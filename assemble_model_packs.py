#!/usr/bin/env python3
"""Assemble provider horizon packs into consensus and multi-model products."""

import argparse
import datetime as dt
import json
import os
import pathlib

import numpy as np

import fetch_weather_data as weather
import validate_consensus as packed
import consensus_pipeline as learning


def geometry_arrays(tile):
    lon0, lat0, dx, dy, channels = tile
    sample = next(iter(channels.values()))
    height, width = sample.shape
    return (
        lon0 + np.arange(width, dtype=np.float32) * dx,
        lat0 + np.arange(height, dtype=np.float32) * dy,
        dx,
        dy,
    )


def components_for_hour(contributors, skill_weights=None, temperature_bias=0.0):
    components = []
    available = set()
    for field_name, channel in weather.MODEL_FIELD_CHANNELS.items():
        if field_name == "precipitation_probability":
            continue
        values = [channels[channel] for _, channels in contributors if channel in channels]
        names = [name for name, channels in contributors if channel in channels]
        families = {weather.MODEL_FAMILIES[name] for name in names}
        if not values or len(families) < 2:
            continue
        stack = np.stack(values)
        weights = weather._family_balanced_weights(names, skill_weights)
        if field_name == "condition_code":
            consensus = weather._condition_consensus(stack, weights)
        elif field_name == "precipitation":
            wet = np.where(np.isfinite(stack), stack >= 0.05, False)
            weighted = weights.reshape((-1, 1, 1))
            total = np.sum(np.where(np.isfinite(stack), weighted, 0), axis=0)
            probability = np.divide(
                np.sum(np.where(wet, weighted, 0), axis=0), total,
                out=np.zeros(stack.shape[1:], dtype=np.float32), where=total > 0,
            )
            wet_amount = weather._weighted_median(np.where(wet, stack, np.nan), weights)
            consensus = np.where(
                probability >= 0.5, np.nan_to_num(wet_amount, nan=0.0), 0.0
            ).astype(np.float32)
            components.append((
                weather.MODEL_FIELD_CHANNELS["precipitation_probability"],
                weather.prepare_grid_values(probability, 3, preserve_missing=True),
            ))
        else:
            consensus = weather._weighted_median(stack, weights)
        if field_name == "temperature" and temperature_bias:
            consensus = consensus + temperature_bias
        components.append((
            channel,
            weather.prepare_grid_values(consensus, 3, preserve_missing=True),
        ))
        available.add(channel)
    return components, available


def regional_learning(state, latitude, longitude, forecast_hour, reference_time):
    region = learning.region_key(latitude, longitude)
    regional = state.get("regions", {}).get(region, {})
    bucket = learning.lead_bucket(forecast_hour)
    weights = regional.get("lead_buckets", {}).get(bucket, {}).get("weights")
    if not weights:
        weights = state.get("weights", {})
    reference = learning.parse_time(reference_time)
    valid = reference + dt.timedelta(hours=int(forecast_hour))
    local_hour = (valid.hour + longitude / 15.0) % 24
    period = "night" if local_hour >= 21 or local_hour < 6 else "day"
    bias = regional.get("temperature_bias", {}).get(period, {})
    bias_value = float(bias.get("value", 0.0)) if int(
        bias.get("samples", 0)
    ) >= 3 else 0.0
    return weights, bias_value


def assemble(root, reference_time, learning_state=None):
    root = pathlib.Path(root)
    output = root / "consensus"
    multi_output = root / "multi"
    output.mkdir(exist_ok=True)
    multi_output.mkdir(exist_ok=True)
    providers = [
        name for name in weather.MODEL_PROVIDER_IDS if (root / name).is_dir()
    ]
    learning_state = learning_state or {}
    tile_names = sorted({
        path.name for name in providers for path in (root / name).glob("*.gpack")
    })
    consensus_hours = set()
    consensus_channels = set()
    for tile_name in tile_names:
        documents = {
            name: packed.read_pack(root / name / tile_name)
            for name in providers if (root / name / tile_name).exists()
        }
        hour_entries = []
        all_hours = sorted({hour for document in documents.values() for hour in document})
        for hour in all_hours:
            contributors = [
                (name, document[hour][4])
                for name, document in documents.items() if hour in document
            ]
            example_tile = next(
                document[hour] for document in documents.values() if hour in document
            )
            geometry = geometry_arrays(example_tile)
            lon_vals, lat_vals, dx, dy = geometry
            skill_weights, temperature_bias = regional_learning(
                learning_state,
                float(np.mean(lat_vals)),
                float(np.mean(lon_vals)),
                int(hour),
                reference_time,
            )
            components, channels = components_for_hour(
                contributors, skill_weights, temperature_bias
            )
            if not components:
                continue
            hour_entries.append((hour, weather.build_binary_tile_bytes(
                lon_vals, lat_vals, dx, dy, components
            )))
            consensus_hours.add(int(hour))
            consensus_channels.update(channels)
        if hour_entries:
            weather.write_tile_pack(output / tile_name, hour_entries)

        day_entries = []
        for day in range(15):
            components = []
            geometry = None
            for name in weather.MULTI_FORECAST_MODELS:
                document = documents.get(name, {})
                temperatures = []
                for hour, tile in document.items():
                    if int(hour) // 24 != day:
                        continue
                    temperature = tile[4].get(weather.MODEL_FIELD_CHANNELS["temperature"])
                    if temperature is not None:
                        temperatures.append(temperature)
                        geometry = geometry_arrays(tile)
                if temperatures:
                    stack = np.stack(temperatures)
                    components.extend([
                        (weather.MULTI_FORECAST_HIGH_CHANNELS[name], weather.prepare_grid_values(np.nanmax(stack, axis=0), 1, preserve_missing=True)),
                        (weather.MULTI_FORECAST_LOW_CHANNELS[name], weather.prepare_grid_values(np.nanmin(stack, axis=0), 1, preserve_missing=True)),
                    ])
            if components and geometry:
                lon_vals, lat_vals, dx, dy = geometry
                day_entries.append((str(day), weather.build_binary_tile_bytes(
                    lon_vals, lat_vals, dx, dy, components
                )))
        if day_entries:
            weather.write_tile_pack(multi_output / tile_name, day_entries)

    models = {}
    for name in providers:
        fragment = root / f"{name}.json"
        if fragment.exists():
            models[name] = json.loads(fragment.read_text())
    if consensus_hours:
        models["consensus"] = {
            "forecast_hours": sorted(consensus_hours),
            "day_count": max(consensus_hours) // 24 + 1,
            "pack_layout": "complete_horizon",
            "fields": [
                field for field, channel in weather.MODEL_FIELD_CHANNELS.items()
                if channel in consensus_channels
            ],
            "model_families": weather.MODEL_FAMILIES,
            "minimum_independent_families": 2,
        }
    manifest = {
        "version": 2,
        "ref_time": reference_time,
        "condition_codes": weather.CONDITION_CODES,
        "field_channels": weather.MODEL_FIELD_CHANNELS,
        "models": models,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--reference-time", required=True)
    parser.add_argument("--weights")
    args = parser.parse_args()
    learning_state = {}
    if args.weights and os.path.exists(args.weights):
        with open(args.weights) as file:
            learning_state = json.load(file)
    assemble(args.models_dir, args.reference_time, learning_state)


if __name__ == "__main__":
    main()
