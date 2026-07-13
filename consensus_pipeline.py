#!/usr/bin/env python3
"""Build observation-trained provider weights and a consensus run manifest.

Provider fetch jobs write normalized JSON independently. This finalizer is kept
small so it can run hourly after METAR observations arrive without downloading
the source model grids again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import tempfile
import urllib.parse
import urllib.request

from weather_time import parse_utc


SCHEMA_VERSION = 2
DEFAULT_ALPHA = 0.15
DEFAULT_SCORE = 1.0
MIN_WEIGHT = 0.02
HISTORY_LIMIT = 48
REGION_SIZE = 20
AVIATION_WEATHER_URL = "https://aviationweather.gov/api/data/metar"


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def parse_time(value):
    return parse_utc(value)


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as file:
            json.dump(value, file, separators=(",", ":"), sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fetch_metar(stations, hours=2):
    """Fetch decoded METAR observations in one request for many stations."""
    query = urllib.parse.urlencode({
        "ids": ",".join(sorted(set(stations))),
        "format": "json",
        "hours": hours,
    })
    request = urllib.request.Request(
        f"{AVIATION_WEATHER_URL}?{query}",
        headers={"User-Agent": "Gusty Weather consensus/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def load_json(path, default=None):
    try:
        with open(path) as file:
            return json.load(file)
    except FileNotFoundError:
        return default


def observation_values(report):
    temperature = report.get("temp")
    wind_speed = report.get("wspd")
    pressure = report.get("pressure_hpa")
    if pressure is None and report.get("altimeter_in_hg") is not None:
        pressure = float(report["altimeter_in_hg"]) * 33.8638866667
    return {
        "temperature_c": float(temperature) if temperature is not None else None,
        "wind_kmh": float(wind_speed) * 1.852 if wind_speed is not None else None,
        "dewpoint_c": report.get("dewpoint"),
        "pressure_hpa": pressure,
        "precipitation_mm": (
            float(report["precipitation_in"]) * 25.4
            if report.get("precipitation_in") is not None else None
        ),
    }


def forecast_error(forecast, observation):
    """Dimensionless error; 1 point is roughly 2 C or 10 km/h."""
    errors = []
    temperature = forecast.get("temperature_c")
    if temperature is not None and observation["temperature_c"] is not None:
        errors.append(abs(float(temperature) - observation["temperature_c"]) / 2.0)
    wind = forecast.get("wind_kmh")
    if wind is not None and observation["wind_kmh"] is not None:
        errors.append(abs(float(wind) - observation["wind_kmh"]) / 10.0)
    dewpoint = forecast.get("dewpoint_c")
    if dewpoint is not None and observation["dewpoint_c"] is not None:
        errors.append(abs(float(dewpoint) - float(observation["dewpoint_c"])) / 3.0)
    pressure = forecast.get("pressure_hpa")
    if pressure is not None and observation["pressure_hpa"] is not None:
        errors.append(abs(float(pressure) - float(observation["pressure_hpa"])) / 4.0)
    precipitation = forecast.get("precipitation_mm")
    if precipitation is not None and observation["precipitation_mm"] is not None:
        predicted_wet = float(precipitation) >= 0.05
        observed_wet = float(observation["precipitation_mm"]) >= 0.1
        errors.append(float(predicted_wet != observed_wet))
    return sum(errors) / len(errors) if errors else None


def region_key(latitude, longitude):
    # Match the model-pack boundaries so every packed tile uses one skill region.
    lat = max(-90, min(70, math.floor((float(latitude) + 90) / REGION_SIZE)
                           * REGION_SIZE - 90))
    lon = max(-180, min(160, math.floor((float(longitude) + 180) / REGION_SIZE)
                            * REGION_SIZE - 180))
    return f"{lat:+03d}_{lon:+04d}"


def lead_bucket(hours):
    hours = int(hours or 0)
    if hours <= 24:
        return "0-24"
    if hours <= 72:
        return "25-72"
    if hours <= 168:
        return "73-168"
    return "169+"


def is_holdout(station):
    return hashlib.sha256(station.encode()).digest()[0] < 26


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (
        ordered[middle - 1] + ordered[middle]
    ) / 2.0


def update_scores(previous, samples, alpha=DEFAULT_ALPHA):
    scores = dict(previous.get("scores", {}))
    counts = dict(previous.get("sample_counts", {}))
    histories = {
        name: list(values) for name, values in previous.get("error_history", {}).items()
    }
    for provider, errors in samples.items():
        valid = [value for value in errors if value is not None and math.isfinite(value)]
        if not valid:
            continue
        error = sum(valid) / len(valid)
        old = float(scores.get(provider, DEFAULT_SCORE))
        scores[provider] = alpha * error + (1.0 - alpha) * old
        counts[provider] = int(counts.get(provider, 0)) + len(valid)
        histories[provider] = (histories.get(provider, []) + valid)[-HISTORY_LIMIT:]
    factors = {}
    for name, score in scores.items():
        robust = median(histories.get(name, [score]))
        effective = 0.5 * score + 0.5 * robust
        factors[name] = max(0.01, math.exp(-effective))
    total = sum(factors.values())
    weights = {
        name: max(MIN_WEIGHT, value / total) for name, value in factors.items()
    }
    normalized_total = sum(weights.values())
    weights = {name: value / normalized_total for name, value in weights.items()}
    return {
        "schema": SCHEMA_VERSION,
        "updated_at": utc_now().isoformat().replace("+00:00", "Z"),
        "alpha": alpha,
        "scores": scores,
        "weights": weights,
        "sample_counts": counts,
        "error_history": histories,
    }


def nearest_forecast(rows, station, observed_at, tolerance=dt.timedelta(minutes=90)):
    candidates = [row for row in rows if row.get("station") == station]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda row: abs(parse_time(row["valid_time"]) - observed_at))
    return nearest if abs(parse_time(nearest["valid_time"]) - observed_at) <= tolerance else None


def _update_bias(previous, errors, alpha=0.2):
    value = float(previous.get("value", 0.0))
    samples = int(previous.get("samples", 0))
    for error in errors:
        value = (1.0 - alpha) * value + alpha * error
        samples += 1
    return {"value": round(value, 3), "samples": samples}


def score_providers(provider_files, observations, previous, alpha):
    reports = {}
    for report in observations:
        station = report.get("icaoId") or report.get("station")
        observed_at = report.get("obsTime") or report.get("reportTime")
        if station and observed_at:
            reports[station] = {
                "time": parse_time(observed_at),
                "values": observation_values(report),
                "latitude": report.get("latitude"),
                "longitude": report.get("longitude"),
            }

    documents = {
        (document.get("provider") or pathlib.Path(path).stem): document
        for path in provider_files for document in [load_json(path, {})]
    }
    global_samples = {}
    regional_samples = {}
    holdout = {}
    bias_errors = {}
    for station, report in reports.items():
        if report["latitude"] is None or report["longitude"] is None:
            continue
        region = region_key(report["latitude"], report["longitude"])
        station_forecasts = {}
        holdout_station = is_holdout(station)
        for provider, document in documents.items():
            forecast = nearest_forecast(
                document.get("forecasts", []), station, report["time"]
            )
            if not forecast:
                continue
            station_forecasts[provider] = forecast
            error = forecast_error(forecast, report["values"])
            if error is None:
                continue
            bucket = lead_bucket(forecast.get("lead_hours"))
            target = holdout if holdout_station else regional_samples
            target.setdefault(region, {}).setdefault(bucket, {}).setdefault(
                provider, []
            ).append(error)
            if not holdout_station:
                global_samples.setdefault(provider, []).append(error)
        if holdout_station and len(station_forecasts) >= 2:
            fields = (
                "temperature_c", "wind_kmh", "dewpoint_c", "pressure_hpa",
                "precipitation_mm",
            )
            blended = {
                field: median(values) if values else None
                for field in fields
                for values in [[
                    float(row[field]) for row in station_forecasts.values()
                    if row.get(field) is not None
                ]]
            }
            error = forecast_error(blended, report["values"])
            if error is not None:
                buckets = {
                    lead_bucket(row.get("lead_hours"))
                    for row in station_forecasts.values()
                }
                for bucket in buckets:
                    holdout.setdefault(region, {}).setdefault(bucket, {}).setdefault(
                        "consensus", []
                    ).append(error)
        temperatures = [
            float(row["temperature_c"]) for row in station_forecasts.values()
            if row.get("temperature_c") is not None
        ]
        actual = report["values"]["temperature_c"]
        if not holdout_station and actual is not None and len(temperatures) >= 2:
            local_hour = (report["time"].hour + float(report["longitude"]) / 15) % 24
            period = "night" if local_hour >= 21 or local_hour < 6 else "day"
            bias_errors.setdefault(region, {}).setdefault(period, []).append(
                float(actual) - median(temperatures)
            )

    global_state = update_scores(previous, global_samples, alpha)
    regions = {}
    previous_regions = previous.get("regions", {})
    for region in set(previous_regions) | set(regional_samples) | set(bias_errors):
        old_region = previous_regions.get(region, {})
        buckets = {}
        for bucket in set(old_region.get("lead_buckets", {})) | set(
            regional_samples.get(region, {})
        ):
            buckets[bucket] = update_scores(
                old_region.get("lead_buckets", {}).get(bucket, {}),
                regional_samples.get(region, {}).get(bucket, {}), alpha,
            )
        old_bias = old_region.get("temperature_bias", {})
        regions[region] = {
            "lead_buckets": buckets,
            "temperature_bias": {
                period: _update_bias(
                    old_bias.get(period, {}), bias_errors.get(region, {}).get(period, [])
                ) for period in ("day", "night")
            },
        }
    global_state["regions"] = regions
    global_state["validation"] = {
        region: {
            bucket: {
                provider: round(sum(values) / len(values), 4)
                for provider, values in providers.items() if values
            } for bucket, providers in buckets.items()
        } for region, buckets in holdout.items()
    }
    global_state["holdout_station_count"] = sum(
        1 for station in reports if is_holdout(station)
    )
    return global_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-file", action="append", default=[])
    parser.add_argument("--stations-file", required=True)
    parser.add_argument("--observations")
    parser.add_argument("--previous-weights")
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    stations_document = load_json(args.stations_file, [])
    stations = [item["icao"] if isinstance(item, dict) else item for item in stations_document]
    observations = load_json(args.observations) if args.observations else fetch_metar(stations)
    previous = load_json(args.previous_weights, {}) if args.previous_weights else {}
    result = score_providers(args.provider_file, observations, previous or {}, args.alpha)
    result["observation_count"] = len(observations)
    result["providers"] = sorted(result["weights"])
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
