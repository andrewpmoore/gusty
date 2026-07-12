#!/usr/bin/env python3
"""Score a generated consensus pack against location observations."""

import argparse
import csv
import datetime as dt
import json
import math
import struct

import numpy as np

import fetch_weather_data as weather


def read_pack(path):
    data = open(path, "rb").read()
    if data[:4] != weather.PACK_MAGIC:
        raise ValueError("Invalid gpack")
    count = struct.unpack_from("<H", data, 5)[0]
    offset = 7
    table = []
    for _ in range(count):
        key_length = data[offset]
        offset += 1
        key = data[offset:offset + key_length].decode("utf-8")
        offset += key_length
        start, length = struct.unpack_from("<II", data, offset)
        offset += 8
        table.append((key, start, length))
    return {
        key: parse_tile(data[offset + start:offset + start + length])
        for key, start, length in table
    }


def parse_tile(tile):
    if tile[:4] != weather.MAGIC:
        raise ValueError("Invalid gtile")
    _, _, count, _, width, height, lon0, lat0, dx, dy = struct.unpack_from(
        "<BBBBHHffff", tile, 4
    )
    metadata = 28
    values_offset = metadata + count * 10
    value_count = width * height
    channels = {}
    for index in range(count):
        channel, scale, value_offset = struct.unpack_from(
            "<hff", tile, metadata + index * 10
        )
        packed = np.frombuffer(
            tile, dtype="<i2", count=value_count,
            offset=values_offset + index * value_count * 2,
        )
        values = packed.astype(np.float32) * scale + value_offset
        values[packed == -32768] = np.nan
        channels[channel] = values.reshape(height, width)
    return lon0, lat0, dx, dy, channels


def sample(tile, latitude, longitude, channel):
    lon0, lat0, dx, dy, channels = tile
    grid = channels.get(channel)
    if grid is None:
        return None
    x = np.clip((longitude - lon0) / dx, 0, grid.shape[1] - 1)
    y = np.clip((latitude - lat0) / dy, 0, grid.shape[0] - 1)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, grid.shape[1] - 1), min(y0 + 1, grid.shape[0] - 1)
    values = np.array([grid[y0, x0], grid[y0, x1], grid[y1, x0], grid[y1, x1]])
    weights = np.array([(1-x+x0)*(1-y+y0), (x-x0)*(1-y+y0),
                        (1-x+x0)*(y-y0), (x-x0)*(y-y0)])
    valid = np.isfinite(values)
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid])) \
        if np.any(valid) and np.sum(weights[valid]) > 0 else None


def parse_time(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    reference = parse_time(manifest["ref_time"])
    tiles = read_pack(args.pack)
    observations = []
    with open(args.observations, newline="") as file:
        for row in csv.DictReader(file):
            row["time"] = parse_time(row["time"])
            observations.append(row)

    errors = {"temperature": [], "wind": [], "rain_brier": [], "condition": []}
    biases = {"temperature": [], "wind": []}
    matched = 0
    codes = {value: key for key, value in weather.CONDITION_CODES.items()}
    for hour_text, tile in tiles.items():
        valid_time = reference + dt.timedelta(hours=int(hour_text))
        nearest = min(observations, key=lambda row: abs(row["time"] - valid_time), default=None)
        if nearest is None or abs(nearest["time"] - valid_time) > dt.timedelta(minutes=90):
            continue
        matched += 1
        temperature = sample(tile, args.lat, args.lon, 0)
        if temperature is not None and nearest.get("temperature_c"):
            difference = temperature - 273.15 - float(nearest["temperature_c"])
            errors["temperature"].append(abs(difference))
            biases["temperature"].append(difference)
        u, v = sample(tile, args.lat, args.lon, 2), sample(tile, args.lat, args.lon, 3)
        if u is not None and v is not None and nearest.get("wind_kmh"):
            difference = math.hypot(u, v) * 3.6 - float(nearest["wind_kmh"])
            errors["wind"].append(abs(difference))
            biases["wind"].append(difference)
        probability = sample(tile, args.lat, args.lon, 12)
        if probability is not None and nearest.get("precipitation_mm"):
            observed_wet = float(nearest["precipitation_mm"]) >= 0.1
            errors["rain_brier"].append((probability - float(observed_wet)) ** 2)
        condition = sample(tile, args.lat, args.lon, 11)
        if condition is not None and nearest.get("condition"):
            errors["condition"].append(
                float(codes.get(round(condition), "Unknown") == nearest["condition"])
            )

    report = {
        "matched_hours": matched,
        "temperature_mae_c": mean(errors["temperature"]),
        "temperature_bias_c": mean(biases["temperature"]),
        "wind_mae_kmh": mean(errors["wind"]),
        "wind_bias_kmh": mean(biases["wind"]),
        "rain_brier_score": mean(errors["rain_brier"]),
        "condition_accuracy": mean(errors["condition"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
