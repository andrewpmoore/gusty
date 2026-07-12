#!/usr/bin/env python3
"""Extract compact provider forecasts at METAR stations from Gusty packs."""

import argparse
import datetime as dt
import json
import math
import os

import fetch_weather_data as weather
import validate_consensus as tiles


def tile_name(latitude, longitude):
    lat = max(-90, min(70, math.floor((latitude + 90) / 20) * 20 - 90))
    lon = max(-180, min(160, math.floor((longitude + 180) / 20) * 20 - 180))
    lat_label = f"N{abs(lat)}" if lat >= 0 else f"S{abs(lat)}"
    lon_label = f"E{abs(lon)}" if lon >= 0 else f"W{abs(lon)}"
    return f"{lat_label}_{lon_label}"


def value(tile, station, field):
    channel = weather.MODEL_FIELD_CHANNELS[field]
    return tiles.sample(tile, station["latitude"], station["longitude"], channel)


def extract(provider, model_dir, manifest, stations):
    reference = tiles.parse_time(manifest["ref_time"])
    rows = []
    pack_cache = {}
    for station in stations:
        name = tile_name(station["latitude"], station["longitude"])
        path = os.path.join(model_dir, provider, f"{name}.gpack")
        if not os.path.exists(path):
            continue
        packs = pack_cache.setdefault(path, tiles.read_pack(path))
        for hour_text, tile in packs.items():
            temperature = value(tile, station, "temperature")
            wind_u = value(tile, station, "wind_u")
            wind_v = value(tile, station, "wind_v")
            rows.append({
                "station": station["icao"],
                "valid_time": (
                    reference + dt.timedelta(hours=int(hour_text))
                ).isoformat().replace("+00:00", "Z"),
                "temperature_c": (
                    temperature - 273.15 if temperature is not None else None
                ),
                "wind_kmh": (
                    math.hypot(wind_u, wind_v) * 3.6
                    if wind_u is not None and wind_v is not None else None
                ),
            })
    return {"schema": 1, "provider": provider, "forecasts": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-dir", default="public/data/models")
    parser.add_argument("--stations-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(os.path.join(args.model_dir, "manifest.json")) as file:
        manifest = json.load(file)
    with open(args.stations_file) as file:
        stations = json.load(file)
    document = extract(args.provider, args.model_dir, manifest, stations)
    with open(args.output, "w") as file:
        json.dump(document, file, separators=(",", ":"))


if __name__ == "__main__":
    main()
