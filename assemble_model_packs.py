#!/usr/bin/env python3
"""Assemble provider horizon packs into consensus and multi-model products."""

import argparse
import datetime as dt
import json
import os
import pathlib
import re

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
            corrected = consensus + temperature_bias
            finite = np.isfinite(stack)
            lower = np.min(np.where(finite, stack, np.inf), axis=0)
            upper = np.max(np.where(finite, stack, -np.inf), axis=0)
            consensus = np.where(
                np.isfinite(consensus),
                np.minimum(np.maximum(corrected, lower), upper),
                np.nan,
            )
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


def _resample_channel(tile, lon_vals, lat_vals, channel):
    values = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=np.float32)
    for y, latitude in enumerate(lat_vals):
        for x, longitude in enumerate(lon_vals):
            value = packed.sample(tile, float(latitude), float(longitude), channel)
            if value is not None:
                values[y, x] = value
    return values


def _channel_on_geometry(tile, lon_vals, lat_vals, channel):
    source_lon, source_lat, source_dx, source_dy = geometry_arrays(tile)
    source = tile[4][channel]
    if (
        source.shape == (len(lat_vals), len(lon_vals))
        and np.allclose(source_lon, lon_vals)
        and np.allclose(source_lat, lat_vals)
    ):
        return source.copy()
    return _resample_channel(tile, lon_vals, lat_vals, channel)


def _copy_channel_into_grid(target, lon_vals, lat_vals, tile, channel):
    lon0, lat0, dx, dy, channels = tile
    source = channels[channel]
    lon1 = lon0 + dx * (source.shape[1] - 1)
    lat1 = lat0 + dy * (source.shape[0] - 1)
    target_x = np.where(
        (lon_vals >= min(lon0, lon1) - 1e-4)
        & (lon_vals <= max(lon0, lon1) + 1e-4)
    )[0]
    target_y = np.where(
        (lat_vals >= min(lat0, lat1) - 1e-4)
        & (lat_vals <= max(lat0, lat1) + 1e-4)
    )[0]
    if target_x.size == 0 or target_y.size == 0:
        return
    source_x = np.rint((lon_vals[target_x] - lon0) / dx).astype(int)
    source_y = np.rint((lat_vals[target_y] - lat0) / dy).astype(int)
    source_x = np.clip(source_x, 0, source.shape[1] - 1)
    source_y = np.clip(source_y, 0, source.shape[0] - 1)
    target[np.ix_(target_y, target_x)] = source[np.ix_(source_y, source_x)]


def add_compact_model_temperature_channels(root):
    """Inject provider temperatures into existing map packs and overviews."""
    root = pathlib.Path(root)
    temperature_dir = root.parent / "temp"
    if not temperature_dir.is_dir():
        return

    provider_documents = {}

    def provider_tile(provider, tile_key, hour):
        cache_key = (provider, tile_key)
        if cache_key not in provider_documents:
            path = root / provider / f"{tile_key}.gpack"
            provider_documents[cache_key] = (
                packed.read_pack(path) if path.exists() else {}
            )
        return provider_documents[cache_key].get(str(hour))

    for pack_path in temperature_dir.glob("temp_*h_*.gpack"):
        match = re.match(r"temp_(\d+)h_", pack_path.name)
        if not match:
            continue
        hour = int(match.group(1))
        document = packed.read_pack(pack_path)
        detailed_keys = [
            key for key in document if not key.endswith(weather.OVERVIEW_KEY_SUFFIX)
        ]
        augmented_tiles = {}
        output_entries = []

        for tile_key in detailed_keys:
            tile = document[tile_key]
            lon_vals, lat_vals, dx, dy = geometry_arrays(tile)
            components = list(tile[4].items())
            channels = dict(tile[4])
            for provider, output_channel in weather.MAP_TEMPERATURE_MODEL_CHANNELS.items():
                if provider == "gfs" or output_channel in channels:
                    continue
                source = provider_tile(provider, tile_key, hour)
                if source is None or weather.MODEL_FIELD_CHANNELS["temperature"] not in source[4]:
                    continue
                values = _channel_on_geometry(
                    source,
                    lon_vals,
                    lat_vals,
                    weather.MODEL_FIELD_CHANNELS["temperature"],
                )
                components.append((output_channel, values))
                channels[output_channel] = values
            tile_bytes = weather.build_binary_tile_bytes(
                lon_vals, lat_vals, dx, dy, components
            )
            output_entries.append((tile_key, tile_bytes))
            augmented_tiles[tile_key] = (
                float(lon_vals[0]), float(lat_vals[0]), dx, dy, channels
            )

        for tile_key, tile in document.items():
            if not tile_key.endswith(weather.OVERVIEW_KEY_SUFFIX):
                continue
            lon_vals, lat_vals, dx, dy = geometry_arrays(tile)
            components = list(tile[4].items())
            for provider, output_channel in weather.MAP_TEMPERATURE_MODEL_CHANNELS.items():
                if provider == "gfs" or output_channel in tile[4]:
                    continue
                values = np.full(
                    (len(lat_vals), len(lon_vals)), np.nan, dtype=np.float32
                )
                for source in augmented_tiles.values():
                    if output_channel in source[4]:
                        _copy_channel_into_grid(
                            values, lon_vals, lat_vals, source, output_channel
                        )
                components.append((output_channel, values))
            output_entries.append((tile_key, weather.build_binary_tile_bytes(
                lon_vals, lat_vals, dx, dy, components
            )))

        if output_entries:
            weather.write_tile_pack(pack_path, output_entries)


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
        consensus_hour_channels = {}
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
            shape = next(iter(example_tile[4].values())).shape
            consensus_hour_channels[int(hour)] = {
                channel: values.reshape(shape)
                for channel, values in components
            }
            hour_entries.append((hour, weather.build_binary_tile_bytes(
                lon_vals, lat_vals, dx, dy, components
            )))
            consensus_hours.add(int(hour))
            consensus_channels.update(channels)
        if hour_entries:
            weather.write_tile_pack(output / tile_name, hour_entries)

        day_entries = []
        for day in range(15):
            geometry = None
            model_hour_channels = {}
            for name in weather.MULTI_FORECAST_MODELS:
                document = documents.get(name, {})
                hourly_channels = []
                for hour, tile in document.items():
                    if int(hour) // 24 != day:
                        continue
                    hourly_channels.append(tile[4])
                    geometry = geometry_arrays(tile)
                if hourly_channels:
                    model_hour_channels[name] = hourly_channels
            blend_hours = [
                channels for hour, channels in consensus_hour_channels.items()
                if int(hour) // 24 == day
            ]
            if blend_hours:
                model_hour_channels["blend"] = blend_hours
            components = weather.multi_forecast_daily_components(
                model_hour_channels
            )
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
    add_compact_model_temperature_channels(root)
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
