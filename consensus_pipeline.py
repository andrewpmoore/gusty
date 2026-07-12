#!/usr/bin/env python3
"""Build observation-trained provider weights and a consensus run manifest.

Provider fetch jobs write normalized JSON independently. This finalizer is kept
small so it can run hourly after METAR observations arrive without downloading
the source model grids again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import pathlib
import tempfile
import urllib.parse
import urllib.request


SCHEMA_VERSION = 1
DEFAULT_ALPHA = 0.15
DEFAULT_SCORE = 1.0
MIN_WEIGHT = 0.02
AVIATION_WEATHER_URL = "https://aviationweather.gov/api/data/metar"


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def parse_time(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


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
    return {
        "temperature_c": float(temperature) if temperature is not None else None,
        "wind_kmh": float(wind_speed) * 1.852 if wind_speed is not None else None,
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
    return sum(errors) / len(errors) if errors else None


def update_scores(previous, samples, alpha=DEFAULT_ALPHA):
    scores = dict(previous.get("scores", {}))
    counts = dict(previous.get("sample_counts", {}))
    for provider, errors in samples.items():
        valid = [value for value in errors if value is not None and math.isfinite(value)]
        if not valid:
            continue
        error = sum(valid) / len(valid)
        old = float(scores.get(provider, DEFAULT_SCORE))
        scores[provider] = alpha * error + (1.0 - alpha) * old
        counts[provider] = int(counts.get(provider, 0)) + len(valid)
    inverse = {name: 1.0 / max(score, 0.05) for name, score in scores.items()}
    total = sum(inverse.values())
    weights = {
        name: max(MIN_WEIGHT, value / total) for name, value in inverse.items()
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
    }


def nearest_forecast(rows, station, observed_at, tolerance=dt.timedelta(minutes=90)):
    candidates = [row for row in rows if row.get("station") == station]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda row: abs(parse_time(row["valid_time"]) - observed_at))
    return nearest if abs(parse_time(nearest["valid_time"]) - observed_at) <= tolerance else None


def score_providers(provider_files, observations, previous, alpha):
    reports = {}
    for report in observations:
        station = report.get("icaoId") or report.get("station")
        observed_at = report.get("obsTime") or report.get("reportTime")
        if station and observed_at:
            reports[station] = (parse_time(observed_at), observation_values(report))

    def score_file(path):
        document = load_json(path, {})
        provider = document.get("provider") or pathlib.Path(path).stem
        errors = []
        for station, (observed_at, values) in reports.items():
            forecast = nearest_forecast(document.get("forecasts", []), station, observed_at)
            if forecast:
                errors.append(forecast_error(forecast, values))
        return provider, errors

    samples = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for provider, errors in executor.map(score_file, provider_files):
            samples[provider] = errors
    return update_scores(previous, samples, alpha)


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
