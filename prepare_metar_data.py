#!/usr/bin/env python3
"""Normalize Aviation Weather Center's full METAR cache for consensus scoring."""

import argparse
import csv
import gzip
import json


def first(row, *names):
    for name in names:
        value = row.get(name)
        if value not in (None, "", "M"):
            return value
    return None


def number(value):
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def rows_from_cache(path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as source:
        lines = (line for line in source if not line.startswith("#"))
        yield from csv.DictReader(lines)


def normalize(rows):
    observations = []
    stations = {}
    for row in rows:
        station = first(row, "station_id", "station", "icaoId", "id")
        observed_at = first(
            row, "observation_time", "obsTime", "reportTime", "receipt_time"
        )
        latitude = number(first(row, "latitude", "lat"))
        longitude = number(first(row, "longitude", "lon"))
        if not station or not observed_at or latitude is None or longitude is None:
            continue
        stations[station] = {
            "icao": station,
            "latitude": latitude,
            "longitude": longitude,
        }
        observations.append({
            "icaoId": station,
            "obsTime": observed_at,
            "temp": number(first(row, "temp_c", "temp")),
            "wspd": number(first(row, "wind_speed_kt", "wspd")),
        })
    return observations, sorted(stations.values(), key=lambda item: item["icao"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--stations", required=True)
    args = parser.parse_args()
    observations, stations = normalize(rows_from_cache(args.cache))
    with open(args.observations, "w") as file:
        json.dump(observations, file, separators=(",", ":"))
    with open(args.stations, "w") as file:
        json.dump(stations, file, separators=(",", ":"))
    print(json.dumps({"observations": len(observations), "stations": len(stations)}))


if __name__ == "__main__":
    main()
