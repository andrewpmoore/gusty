import os
import shutil
import requests
import datetime
import time
import json
import argparse
import concurrent.futures
import sys
import struct
import math
import numpy as np
import xarray as xr
import cfgrib

# --- CONFIGURATION REGISTRY ---
NOAA_CONFIG = {
    "wind": {
        "type": "gfs_atmos",
        "vars": {"var_UGRD": "on", "var_VGRD": "on"},
        "level": {"lev_10_m_above_ground": "on"},
        "grid_type": "vector"
    },
    "gusts": {
        "type": "gfs_atmos",
        "vars": {"var_GUST": "on"},
        "level": {"lev_surface": "on"},
        "grid_type": "scalar" # Gust is a single speed value (m/s), not a vector
    },
    "temp": {
        "type": "gfs_atmos",
        "vars": {
            "var_TMP": "on",
            "var_DPT": "on",
            "var_RH": "on",
            "var_UGRD": "on",
            "var_VGRD": "on",
            "var_GUST": "on",
            "var_PRMSL": "on",
            "var_TCDC": "on",
            "var_VIS": "on",
            "var_APCP": "on",
        },
        "level": {
            "lev_2_m_above_ground": "on",
            "lev_10_m_above_ground": "on",
            "lev_surface": "on",
            "lev_mean_sea_level": "on",
            "lev_entire_atmosphere": "on",
        },
        "grid_type": "scalar"
    },
    "humidity": {
        "type": "gfs_atmos",
        "vars": {"var_RH": "on"},
        "level": {"lev_2_m_above_ground": "on"},
        "grid_type": "scalar"
    },
    "clouds": {
        "type": "gfs_atmos",
        "vars": {"var_TCDC": "on"},
        "level": {"lev_entire_atmosphere": "on"},
        "grid_type": "scalar"
    },
    "visibility": {
        "type": "gfs_atmos",
        "vars": {"var_VIS": "on"},
        "level": {"lev_surface": "on"},
        "grid_type": "scalar"
    },
    "pressure": {
        "type": "gfs_atmos",
        "vars": {"var_PRMSL": "on"},
        "level": {"lev_mean_sea_level": "on"},
        "grid_type": "scalar"
    },
    "waves": {
        "type": "direct_download", 
        "vars": {}, 
        "level": {},
        "grid_type": "vector"
    },
    "smoke": {
        "type": "gefs_chem_idx",
        "grid_type": "scalar",
        "idx_match": [":PMTF:surface:", "aerosol=Total Aerosol", "aerosol_size <2.5e-06"]
    },
    "air_quality": {
        "type": "gefs_chem_idx",
        "grid_type": "scalar",
        "idx_match": [":PMTF:surface:", "aerosol=Total Aerosol", "aerosol_size <2.5e-06"]
    },
    "dust": {
        "type": "gefs_chem_idx",
        "grid_type": "scalar",
        "idx_match": [":PMTF:surface:", "aerosol=Dust Dry", "aerosol_size <2.5e-06"]
    },
    "snow": {
        "type": "gfs_atmos",
        "vars": {"var_SNOD": "on"},
        "level": {"lev_surface": "on"},
        "grid_type": "scalar"
    },
    "cape": {
        "type": "gfs_atmos",
        "vars": {"var_CAPE": "on"},
        "level": {"lev_surface": "on"},
        "grid_type": "scalar"
    },
    "storm_potential": {
        "type": "derived_storm_potential",
        "grid_type": "scalar",
        "sources": {
            "cape": {
                "idx_match": [":CAPE:surface:"]
            },
            "composite_reflectivity": {
                "idx_match": [":REFC:entire atmosphere:"]
            },
            "cin": {
                "idx_match": [":CIN:surface:"],
                "optional": True
            },
            "precipitation_rate": {
                "idx_match": [":PRATE:surface:"],
                "optional": True
            },
            "convective_precipitation_rate": {
                "idx_match": [":CPRAT:surface:"],
                "optional": True
            },
            "precipitable_water": {
                "idx_match": [":PWAT:entire atmosphere"],
                "optional": True
            },
            "dewpoint": {
                "idx_match": [":DPT:2 m above ground:"],
                "optional": True
            },
            "wind_u_500": {
                "idx_match": [":UGRD:500 mb:"],
                "optional": True
            },
            "wind_v_500": {
                "idx_match": [":VGRD:500 mb:"],
                "optional": True
            },
            "wind_u_850": {
                "idx_match": [":UGRD:850 mb:"],
                "optional": True
            },
            "wind_v_850": {
                "idx_match": [":VGRD:850 mb:"],
                "optional": True
            }
        },
        "metadata": {
            "units": "index",
            "value_range": [0, 100],
            "description": "Colored storm-potential layer derived from GFS instability, modeled convection, moisture, cap strength, and deep-layer shear.",
            "source_fields": [
                "CAPE_surface",
                "REFC_entire_atmosphere",
                "CIN_surface_optional",
                "PRATE_surface_optional",
                "CPRAT_surface_optional",
                "PWAT_entire_atmosphere_optional",
                "DPT_2m_optional",
                "UGRD/VGRD_500mb_optional",
                "UGRD/VGRD_850mb_optional"
            ],
            "formula": "Instability blended with reflectivity/precipitation overlap, moisture support, CIN suppression, and optional 850-500 mb shear boost; scaled to 0-100"
        }
    },
    "storm_motion": {
        "type": "derived_storm_potential",
        "grid_type": "vector",
        "sources": {
            "cape": {
                "idx_match": [":CAPE:surface:"]
            },
            "composite_reflectivity": {
                "idx_match": [":REFC:entire atmosphere:"]
            },
            "cin": {
                "idx_match": [":CIN:surface:"],
                "optional": True
            },
            "precipitation_rate": {
                "idx_match": [":PRATE:surface:"],
                "optional": True
            },
            "convective_precipitation_rate": {
                "idx_match": [":CPRAT:surface:"],
                "optional": True
            },
            "precipitable_water": {
                "idx_match": [":PWAT:entire atmosphere"],
                "optional": True
            },
            "dewpoint": {
                "idx_match": [":DPT:2 m above ground:"],
                "optional": True
            },
            "wind_u_500": {
                "idx_match": [":UGRD:500 mb:"],
                "optional": True
            },
            "wind_v_500": {
                "idx_match": [":VGRD:500 mb:"],
                "optional": True
            },
            "wind_u_850": {
                "idx_match": [":UGRD:850 mb:"],
                "optional": True
            },
            "wind_v_850": {
                "idx_match": [":VGRD:850 mb:"],
                "optional": True
            },
            "wind_u": {
                "idx_match": [":UGRD:700 mb:"]
            },
            "wind_v": {
                "idx_match": [":VGRD:700 mb:"]
            }
        },
        "metadata": {
            "units": "m/s weighted by storm potential",
            "description": "Optional vector storm-motion layer for particles/arrows, using 700 mb steering wind weighted by the storm-potential score.",
            "source_fields": [
                "CAPE_surface",
                "REFC_entire_atmosphere",
                "optional storm-potential ingredients",
                "UGRD_700mb",
                "VGRD_700mb"
            ],
            "formula": "U/V steering wind multiplied by a 0-1 storm score from the storm-potential formula"
        }
    }
}

OUTPUT_DIR = "public/data"
HOURS_TO_FETCH = [0, 3, 6, 9, 12, 15, 18, 21, 24, 36, 48, 72]
# Temperature comparisons need enough samples to derive each model's daily
# high and low at the user's location. Keep three-hour resolution through the
# ICON-EU horizon, then use the native six-hour cadence of the global AI
# models through 14 days.
TEMPERATURE_HOURS_TO_FETCH = [
    *range(0, 121, 3),
    *range(126, 337, 6),
]
TILE_SIZE = 20
PACK_LAT_SIZE = 60
PACK_LON_SIZE = 80
TILE_FORMAT = {
    "name": "gusty-grid",
    "version": 1,
    "extension": "gtile",
    "encoding": "int16-quantized",
    "endianness": "little",
    "missing_value": -32768,
    "decode_formula": "value = offset + raw * scale"
}
PACK_FORMAT = {
    "name": "gusty-pack",
    "version": 1,
    "extension": "gpack",
    "lat_size": PACK_LAT_SIZE,
    "lon_size": PACK_LON_SIZE,
    "contains": TILE_FORMAT["extension"],
    "overview_key_suffix": "_60",
    "overview_sample_step": 4,
    "offset_base": "start_of_data_section"
}

MAGIC = b"GSTY"
VERSION = 1
ENCODING_INT16_QUANTIZED = 1
MISSING_VALUE = -32768
INT16_MIN_VALUE = -32767
INT16_MAX_VALUE = 32767
PACK_MAGIC = b"GPAK"
PACK_VERSION = 1
OVERVIEW_KEY_SUFFIX = "_60"
OVERVIEW_SAMPLE_STEP = 4

# Stable IDs shared with Gusty's condition-code vocabulary. Two four-bit IDs
# are packed into each categorical tile channel.
CONDITION_CODES = {
    "Unknown": 0,
    "Clear": 1,
    "MostlyClear": 2,
    "PartlyCloudy": 3,
    "MostlyCloudy": 4,
    "Cloudy": 5,
    "Drizzle": 6,
    "Rain": 7,
    "HeavyRain": 8,
    "Snow": 9,
    "HeavySnow": 10,
    "Sleet": 11,
    "FreezingRain": 12,
    "Thunderstorms": 13,
    "Foggy": 14,
    "Hail": 15,
}

CONDITION_CHANNEL_PAIRS = {
    30: ("ifs", "aifs"),
    31: ("aigfs", "icon"),
}

MODEL_FIELD_CHANNELS = {
    "temperature": 0,
    "dewpoint": 1,
    "wind_u": 2,
    "wind_v": 3,
    "wind_gust": 4,
    "pressure": 5,
    "humidity": 6,
    "cloud_cover": 7,
    "precipitation": 8,
    "snowfall": 9,
    "visibility": 10,
    "condition_code": 11,
    "precipitation_probability": 12,
}

MODEL_FAMILIES = {
    "gfs": "noaa",
    "aigfs": "noaa",
    "ifs": "ecmwf",
    "aifs": "ecmwf",
    "icon": "dwd",
}
CONSENSUS_CONFIDENCE_CHANNELS = {
    field: 40 + index for index, field in enumerate(MODEL_FIELD_CHANNELS)
}
CONSENSUS_SPREAD_CHANNELS = {
    field: 60 + index for index, field in enumerate(MODEL_FIELD_CHANNELS)
}
CONSENSUS_PARTICIPATION_CHANNELS = {
    field: 80 + index for index, field in enumerate(MODEL_FIELD_CHANNELS)
}
CONSENSUS_QUALITY_FIELDS = {"temperature", "precipitation", "condition_code"}

MODEL_PROVIDER_IDS = ("gfs", "ifs", "aifs", "aigfs", "icon")
MULTI_FORECAST_MODELS = ("ifs", "aifs", "aigfs", "icon")
MULTI_FORECAST_HIGH_CHANNELS = {
    model: 20 + index for index, model in enumerate(MULTI_FORECAST_MODELS)
}
MULTI_FORECAST_LOW_CHANNELS = {
    model: 24 + index for index, model in enumerate(MULTI_FORECAST_MODELS)
}
MODEL_WORK_DIR = os.path.join(OUTPUT_DIR, ".model_provider_work")
MODEL_PREVIOUS_ACCUMULATIONS = {}
CONSENSUS_WEIGHTS_FILE = os.environ.get(
    "CONSENSUS_WEIGHTS_FILE", os.path.join(OUTPUT_DIR, "models", "weights.json")
)

def clean_output_directory():
    """Wipes the output directory."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_run_time():
    now = datetime.datetime.now(datetime.timezone.utc)
    possible_runs = [0, 6, 12, 18]
    check_time = now - datetime.timedelta(hours=4, minutes=30) 
    
    date_str = check_time.strftime("%Y%m%d")
    hour = check_time.hour
    run_hour = max([h for h in possible_runs if h <= hour], default=18)
    if run_hour == 18 and hour < 4:
        prev_day = check_time - datetime.timedelta(days=1)
        date_str = prev_day.strftime("%Y%m%d")
    return date_str, f"{run_hour:02d}"

def coordinate_label(lat_start, lon_start):
    lat_label = f"N{abs(lat_start)}" if lat_start >= 0 else f"S{abs(lat_start)}"
    lon_label = f"E{abs(lon_start)}" if lon_start >= 0 else f"W{abs(lon_start)}"
    return lat_label, lon_label

def tile_key(lat_start, lon_start):
    lat_label, lon_label = coordinate_label(lat_start, lon_start)
    return f"{lat_label}_{lon_label}"

def tile_filename(job_type, forecast_hour, lat_start, lon_start):
    return f"{job_type}_{forecast_hour}h_{tile_key(lat_start, lon_start)}.{TILE_FORMAT['extension']}"

def pack_origin(lat_start, lon_start):
    pack_lat_start = math.floor(lat_start / PACK_LAT_SIZE) * PACK_LAT_SIZE
    pack_lat_start = max(-90, min(60, pack_lat_start))

    # Anchor longitude packs so W120/W100/W80/W60 share one W120 pack.
    pack_lon_start = math.floor((lon_start + 120) / PACK_LON_SIZE) * PACK_LON_SIZE - 120
    pack_lon_start = max(-180, min(120, pack_lon_start))
    return pack_lat_start, pack_lon_start

def pack_filename(job_type, forecast_hour, lat_start, lon_start):
    lat_label, lon_label = coordinate_label(lat_start, lon_start)
    return f"{job_type}_{forecast_hour}h_{lat_label}_{lon_label}.{PACK_FORMAT['extension']}"

def overview_tile_key(lat_start, lon_start):
    return f"{tile_key(lat_start, lon_start)}{OVERVIEW_KEY_SUFFIX}"

def precision_for_job(job_type):
    return 3 if job_type in ["snow", "smoke", "dust", "air_quality"] else 1

def prepare_grid_values(values, precision, preserve_missing=False):
    filled = values if preserve_missing else np.where(np.isnan(values), 0, values)
    return np.round(filled.astype(np.float32), precision).flatten()

def quantize_int16(values):
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, -32768, dtype="<i2"), 1.0, 0.0
    valid = values[finite]
    min_value = float(np.min(valid))
    max_value = float(np.max(valid))

    if min_value == max_value:
        quantized = np.full(values.shape, -32768, dtype="<i2")
        quantized[finite] = 0
        return quantized, 1.0, min_value

    scale = (max_value - min_value) / (INT16_MAX_VALUE - INT16_MIN_VALUE)
    offset = (max_value + min_value) / 2.0
    quantized = np.full(values.shape, -32768, dtype="<i2")
    quantized[finite] = np.clip(
        np.rint((values[finite] - offset) / scale),
        INT16_MIN_VALUE,
        INT16_MAX_VALUE,
    ).astype("<i2")
    return quantized, float(scale), float(offset)

def build_binary_tile_bytes(lon_vals, lat_vals, dx, dy, components):
    expected_value_count = len(lon_vals) * len(lat_vals)
    header = bytearray()
    header.extend(MAGIC)
    header.extend(struct.pack(
        "<BBBBHHffff",
        VERSION,
        ENCODING_INT16_QUANTIZED,
        len(components),
        0,
        len(lon_vals),
        len(lat_vals),
        float(lon_vals[0]),
        float(lat_vals[0]),
        float(dx),
        float(dy)
    ))

    quantized_components = []
    for parameter_number, values in components:
        if values.size != expected_value_count:
            raise ValueError(
                f"Component {parameter_number} has {values.size} values, "
                f"expected {expected_value_count}"
            )
        quantized, scale, offset = quantize_int16(values)
        header.extend(struct.pack("<hff", int(parameter_number), scale, offset))
        quantized_components.append(quantized)

    output = bytearray(header)
    for quantized in quantized_components:
        output.extend(quantized.tobytes())
    return bytes(output)

def write_binary_tile(path, lon_vals, lat_vals, dx, dy, components):
    tile_bytes = build_binary_tile_bytes(lon_vals, lat_vals, dx, dy, components)
    with open(path, "wb") as f:
        f.write(tile_bytes)

def write_tile_pack(pack_path, entries):
    header = bytearray()
    data = bytearray()
    table = []

    for key, tile_source in entries:
        if isinstance(tile_source, bytes):
            tile_bytes = tile_source
        else:
            with open(tile_source, "rb") as f:
                tile_bytes = f.read()
        offset = len(data)
        data.extend(tile_bytes)
        table.append((key, offset, len(tile_bytes)))

    header.extend(PACK_MAGIC)
    header.extend(struct.pack("<BH", PACK_VERSION, len(table)))

    for key, offset, length in table:
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 255:
            raise ValueError(f"Pack key is too long: {key}")
        header.extend(struct.pack("<B", len(key_bytes)))
        header.extend(key_bytes)
        header.extend(struct.pack("<II", offset, length))

    with open(pack_path, "wb") as f:
        f.write(header)
        f.write(data)

def read_binary_tile(path):
    """Read one generated tile for server-side forecast aggregation."""
    with open(path, "rb") as file:
        tile = file.read()
    if tile[:4] != MAGIC:
        raise ValueError(f"Invalid tile header: {path}")
    _, _, channel_count, _, width, height, lon0, lat0, dx, dy = struct.unpack_from(
        "<BBBBHHffff", tile, 4
    )
    metadata_offset = 28
    values_offset = metadata_offset + channel_count * 10
    value_count = width * height
    channels = {}
    for index in range(channel_count):
        channel, scale, offset = struct.unpack_from(
            "<hff", tile, metadata_offset + index * 10
        )
        start = values_offset + index * value_count * 2
        quantized = np.frombuffer(
            tile, dtype="<i2", count=value_count, offset=start
        ).astype(np.float32)
        decoded = quantized * scale + offset
        decoded[quantized == -32768] = np.nan
        channels[channel] = decoded.reshape(height, width)
    lon_vals = lon0 + np.arange(width, dtype=np.float32) * dx
    lat_vals = lat0 + np.arange(height, dtype=np.float32) * dy
    return lon_vals, lat_vals, dx, dy, channels

def download_idx_message(base_url, idx_match, output_path):
    idx_url = f"{base_url}.idx"
    idx_response = requests.get(idx_url, timeout=60)
    if idx_response.status_code != 200:
        print(f"⚠️ NOAA IDX Error {idx_response.status_code}")
        return False

    rows = []
    for line in idx_response.text.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3: continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        rows.append((line, offset))

    for index, (line, start) in enumerate(rows):
        if all(token in line for token in idx_match):
            end = rows[index + 1][1] - 1 if index + 1 < len(rows) else None
            headers = {"Range": f"bytes={start}-{end if end is not None else ''}"}
            response = requests.get(base_url, headers=headers, stream=True, timeout=180)
            if response.status_code not in (200, 206):
                print(f"⚠️ NOAA Range Error {response.status_code}")
                return False
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=16384):
                    f.write(chunk)
            return True

    print(f"⚠️ NOAA IDX field not found: {' '.join(idx_match)}")
    return False

def make_components(subset, config, job_type, sample_step=1, subset_min=None, subset_max=None, model_grids=None):
    components = []
    data_vars = list(subset.data_vars)

    if config['grid_type'] == 'vector':
        if len(data_vars) < 2:
            return components
        var1, var2 = subset[data_vars[0]], subset[data_vars[1]]
        components.append((2, prepare_grid_values(var1.values[::sample_step, ::sample_step], 1)))
        components.append((3, prepare_grid_values(var2.values[::sample_step, ::sample_step], 1)))
    else:
        if not data_vars:
            return components
        var1 = subset[data_vars[0]]
        components.append((
            0,
            prepare_grid_values(
                var1.values[::sample_step, ::sample_step],
                precision_for_job(job_type)
            )
        ))

        # Add temperature variance components (min/max)
        if job_type == "temp" and subset_min is not None and subset_max is not None:
            components.append((
                10,
                prepare_grid_values(
                    subset_min[::sample_step, ::sample_step],
                    precision_for_job(job_type)
                )
            ))
            components.append((
                11,
                prepare_grid_values(
                    subset_max[::sample_step, ::sample_step],
                    precision_for_job(job_type)
                )
            ))

            # Individual multi-model temperature values.  These channels are
            # kept alongside the aggregate range so clients can show a
            # per-model tooltip without downloading the source GRIB files.
            model_channels = {
                "ifs": 12,
                "aifs": 13,
                "aigfs": 14,
                "icon": 15,
            }
            for model_name, parameter_number in model_channels.items():
                model_grid = model_grids.get(model_name) if model_grids else None
                if model_grid is None:
                    continue
                components.append((
                    parameter_number,
                    prepare_grid_values(
                        model_grid.values[::sample_step, ::sample_step],
                        precision_for_job(job_type)
                    )
                ))

    return components

def build_pack_overview_tile(ds, config, job_type, tile_origins, min_grid=None, max_grid=None, model_grids=None):
    if not tile_origins:
        return None

    lat_starts = [lat_start for lat_start, _ in tile_origins]
    lon_starts = [lon_start for _, lon_start in tile_origins]
    pack_lat_start = min(lat_starts)
    pack_lat_end = min(max(lat_starts) + TILE_SIZE, 90)
    pack_lon_start = min(lon_starts)
    pack_lon_end = min(max(lon_starts) + TILE_SIZE, 180)

    subset = ds.sel(
        latitude=slice(pack_lat_end, pack_lat_start),
        longitude=slice(pack_lon_start, pack_lon_end)
    )
    if subset.latitude.size == 0 or subset.longitude.size == 0:
        return None

    lat_vals = subset.latitude.values[::OVERVIEW_SAMPLE_STEP]
    lon_vals = subset.longitude.values[::OVERVIEW_SAMPLE_STEP]
    if len(lat_vals) == 0 or len(lon_vals) == 0:
        return None

    dy = (lat_vals[-1] - lat_vals[0]) / (len(lat_vals) - 1) if len(lat_vals) > 1 else 1.0
    dx = (lon_vals[-1] - lon_vals[0]) / (len(lon_vals) - 1) if len(lon_vals) > 1 else 1.0

    subset_min, subset_max = None, None
    if min_grid is not None and max_grid is not None:
        subset_min = min_grid.sel(
            latitude=slice(pack_lat_end, pack_lat_start),
            longitude=slice(pack_lon_start, pack_lon_end)
        ).values
        subset_max = max_grid.sel(
            latitude=slice(pack_lat_end, pack_lat_start),
            longitude=slice(pack_lon_start, pack_lon_end)
        ).values

    subset_model_grids = {}
    for model_name, model_grid in (model_grids or {}).items():
        subset_model_grids[model_name] = model_grid.sel(
            latitude=slice(pack_lat_end, pack_lat_start),
            longitude=slice(pack_lon_start, pack_lon_end)
        )

    components = make_components(
        subset,
        config,
        job_type,
        OVERVIEW_SAMPLE_STEP,
        subset_min=subset_min,
        subset_max=subset_max,
        model_grids=subset_model_grids
    )
    if not components:
        return None

    return build_binary_tile_bytes(lon_vals, lat_vals, dx, dy, components)

def normalize_dataset_coordinates(ds):
    if "longitude" not in ds.coords:
        raise ValueError("Dataset does not contain 'longitude' coordinate")
    return ds.assign_coords(
        longitude=(((ds.coords["longitude"] + 180) % 360) - 180)
    ).sortby('longitude')

def _first_variable(datasets, names):
    for ds in datasets:
        for name in names:
            if name in ds.data_vars:
                try:
                    value = ds[name].squeeze(drop=True)
                    if "latitude" in value.dims and "longitude" in value.dims:
                        return value
                except Exception as error:
                    print(f"      ⚠️ Could not read GRIB field {name}: {error}")
    return None

def _aligned_field(field, target):
    if field is None:
        return None
    dataset = normalize_dataset_coordinates(field.to_dataset(name="value"))
    return dataset["value"].interp_like(target, method="nearest").load()

def _precipitation_mm(field):
    if field is None:
        return None
    units = str(field.attrs.get("units", "")).lower()
    # ECMWF accumulated precipitation is metres of water; NOAA commonly uses
    # kg m-2, which is numerically equivalent to millimetres.
    return field * 1000.0 if units in {"m", "metre", "meter"} else field

def _interval_accumulation(model_name, field_name, field, source_attrs):
    start_step = source_attrs.get("GRIB_startStep")
    end_step = source_attrs.get("GRIB_endStep")
    if start_step == 0 and end_step not in (None, 0):
        key = (model_name, field_name)
        previous = MODEL_PREVIOUS_ACCUMULATIONS.get(key)
        MODEL_PREVIOUS_ACCUMULATIONS[key] = field
        if previous is not None:
            return (field - previous).clip(min=0)
    return field

def derive_condition_grid(
    target,
    cloud_cover=None,
    precipitation=None,
    convective_precipitation=None,
    snowfall=None,
    temperature=None,
    dewpoint=None,
):
    """Derive conservative Gusty condition IDs from common model fields."""
    condition = xr.full_like(target, CONDITION_CODES["Unknown"], dtype=np.float32)

    cloud = _aligned_field(cloud_cover, target)
    if cloud is not None:
        cloud = xr.where(cloud > 1.5, cloud / 100.0, cloud).clip(0, 1)
        condition = xr.where(cloud < 0.12, CONDITION_CODES["Clear"], condition)
        condition = xr.where(
            (cloud >= 0.12) & (cloud < 0.32),
            CONDITION_CODES["MostlyClear"],
            condition,
        )
        condition = xr.where(
            (cloud >= 0.32) & (cloud < 0.68),
            CONDITION_CODES["PartlyCloudy"],
            condition,
        )
        condition = xr.where(
            (cloud >= 0.68) & (cloud < 0.88),
            CONDITION_CODES["MostlyCloudy"],
            condition,
        )
        condition = xr.where(cloud >= 0.88, CONDITION_CODES["Cloudy"], condition)

    temp = _aligned_field(temperature, target)
    dew = _aligned_field(dewpoint, target)
    if temp is not None and dew is not None and cloud is not None:
        condition = xr.where(
            (np.abs(temp - dew) <= 1.0) & (cloud >= 0.75),
            CONDITION_CODES["Foggy"],
            condition,
        )

    precip = _precipitation_mm(_aligned_field(precipitation, target))
    convective = _precipitation_mm(
        _aligned_field(convective_precipitation, target)
    )
    snow = _precipitation_mm(_aligned_field(snowfall, target))

    if precip is not None:
        condition = xr.where(
            (precip > 0.01) & (precip < 0.25),
            CONDITION_CODES["Drizzle"],
            condition,
        )
        condition = xr.where(
            (precip >= 0.25) & (precip < 8.0),
            CONDITION_CODES["Rain"],
            condition,
        )
        condition = xr.where(
            precip >= 8.0, CONDITION_CODES["HeavyRain"], condition
        )

    if snow is not None:
        rain_amount = precip - snow if precip is not None else None
        condition = xr.where(
            snow >= 0.05, CONDITION_CODES["Snow"], condition
        )
        condition = xr.where(
            snow >= 5.0, CONDITION_CODES["HeavySnow"], condition
        )
        if rain_amount is not None:
            condition = xr.where(
                (snow >= 0.05) & (rain_amount >= 0.05),
                CONDITION_CODES["Sleet"],
                condition,
            )

    if precip is not None and temp is not None:
        liquid_only = True if snow is None else snow < 0.05
        condition = xr.where(
            (precip >= 0.05) & (temp <= 273.65) &
            liquid_only,
            CONDITION_CODES["FreezingRain"],
            condition,
        )

    if convective is not None:
        condition = xr.where(
            convective >= 0.5, CONDITION_CODES["Thunderstorms"], condition
        )

    return condition.astype(np.float32)

def extract_condition_grid(grib_path, target):
    """Read condition inputs from a model file already downloaded for temp."""
    datasets = []
    try:
        datasets = cfgrib.open_datasets(grib_path, backend_kwargs={"indexpath": ""})
        cloud = _first_variable(datasets, ["tcc", "clct", "TCDC"])
        precip = _first_variable(datasets, ["tp", "tot_prec", "APCP"])
        convective = _first_variable(datasets, ["cp", "rain_con", "ACPCP"])
        snow = _first_variable(datasets, ["sf", "snow_gsp", "ASNOW"])
        temperature = _first_variable(datasets, ["t2m", "2t", "t"])
        dewpoint = _first_variable(datasets, ["d2m", "2d"])
        if all(value is None for value in [cloud, precip, snow]):
            return None
        return derive_condition_grid(
            target,
            cloud_cover=cloud,
            precipitation=precip,
            convective_precipitation=convective,
            snowfall=snow,
            temperature=temperature,
            dewpoint=dewpoint,
        )
    except Exception as error:
        print(f"      ⚠️ Condition extraction failed for {grib_path}: {error}")
        return None
    finally:
        for ds in datasets:
            ds.close()

def extract_provider_fields(model_name, grib_path, target, temperature):
    """Extract normalized provider fields while the source GRIB is local."""
    datasets = []
    try:
        datasets = cfgrib.open_datasets(grib_path, backend_kwargs={"indexpath": ""})
        raw = {
            "dewpoint": _first_variable(datasets, ["d2m", "2d"]),
            "wind_u": _first_variable(datasets, ["u10", "10u", "u"]),
            "wind_v": _first_variable(datasets, ["v10", "10v", "v"]),
            "wind_gust": _first_variable(datasets, ["fg10", "gust", "GUST"]),
            "pressure": _first_variable(datasets, ["msl", "prmsl", "PRMSL"]),
            "humidity": _first_variable(datasets, ["r2", "rh2m", "RH"]),
            "cloud_cover": _first_variable(datasets, ["tcc", "clct", "TCDC"]),
            "precipitation": _first_variable(datasets, ["tp", "tot_prec", "APCP"]),
            "snowfall": _first_variable(datasets, ["sf", "snow_gsp", "ASNOW"]),
            "visibility": _first_variable(datasets, ["vis", "visibility", "VIS"]),
        }
        fields = {"temperature": temperature.load()}
        for name, value in raw.items():
            if value is None:
                continue
            try:
                aligned = _aligned_field(value, target)
                if aligned is not None:
                    if name in {"precipitation", "snowfall"}:
                        aligned = _interval_accumulation(
                            model_name, name, aligned, value.attrs
                        )
                    fields[name] = aligned
            except Exception as error:
                print(
                    f"      ⚠️ {model_name} {name} extraction failed: {error}"
                )

        if "cloud_cover" in fields:
            cloud = fields["cloud_cover"]
            fields["cloud_cover"] = xr.where(cloud > 1.5, cloud / 100.0, cloud).clip(0, 1)
        for name in ("precipitation", "snowfall"):
            if name in fields:
                fields[name] = _precipitation_mm(fields[name])
                fields[name].attrs["units"] = "mm"
        if "humidity" in fields:
            humidity = fields["humidity"]
            fields["humidity"] = xr.where(
                humidity > 1.5, humidity / 100.0, humidity
            ).clip(0, 1)
        elif "dewpoint" in fields:
            temp_c = fields["temperature"] - 273.15
            dew_c = fields["dewpoint"] - 273.15
            fields["humidity"] = (
                np.exp((17.625 * dew_c) / (243.04 + dew_c)) /
                np.exp((17.625 * temp_c) / (243.04 + temp_c))
            ).clip(0, 1)

        try:
            fields["condition_code"] = derive_condition_grid(
                target,
                cloud_cover=fields.get("cloud_cover"),
                precipitation=fields.get("precipitation"),
                convective_precipitation=_first_variable(
                    datasets, ["cp", "rain_con", "ACPCP"]
                ),
                snowfall=fields.get("snowfall"),
                temperature=temperature,
                dewpoint=raw["dewpoint"],
            )
        except Exception as error:
            print(f"      ⚠️ {model_name} condition derivation failed: {error}")
        return fields
    except Exception as error:
        print(f"      ⚠️ Provider field extraction failed for {grib_path}: {error}")
        return {"temperature": temperature.load()}
    finally:
        for ds in datasets:
            ds.close()

def write_model_provider_hour(model_name, forecast_hour, fields):
    """Write temporary per-hour tiles; these are compacted into daily packs."""
    if not fields:
        return
    reference = next(iter(fields.values()))
    model_dir = os.path.join(MODEL_WORK_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    for lat_start in range(-90, 90, TILE_SIZE):
        for lon_start in range(-180, 180, TILE_SIZE):
            lat_end = min(lat_start + TILE_SIZE, 90)
            lon_end = min(lon_start + TILE_SIZE, 180)
            subset = reference.sel(
                latitude=slice(lat_end, lat_start),
                longitude=slice(lon_start, lon_end),
            )
            if subset.latitude.size == 0 or subset.longitude.size == 0:
                continue
            lat_vals = subset.latitude.values
            lon_vals = subset.longitude.values
            dy = ((lat_vals[-1] - lat_vals[0]) / (len(lat_vals) - 1)
                  if len(lat_vals) > 1 else 1.0)
            dx = ((lon_vals[-1] - lon_vals[0]) / (len(lon_vals) - 1)
                  if len(lon_vals) > 1 else 1.0)
            components = []
            for field_name, channel in MODEL_FIELD_CHANNELS.items():
                field = fields.get(field_name)
                if field is None:
                    continue
                values = field.sel(
                    latitude=slice(lat_end, lat_start),
                    longitude=slice(lon_start, lon_end),
                ).values
                components.append((
                    channel,
                    prepare_grid_values(values, 3 if field_name in {
                        "humidity", "precipitation", "snowfall"
                    } else 1, preserve_missing=True),
                ))
            if not components:
                continue
            path = os.path.join(
                model_dir,
                f"{forecast_hour}h_{tile_key(lat_start, lon_start)}.gtile",
            )
            write_binary_tile(path, lon_vals, lat_vals, dx, dy, components)

def compact_multi_forecast_tiles():
    """Precompute every model's daily high/low into one request per tile."""
    output_dir = os.path.join(OUTPUT_DIR, "models", "multi")
    os.makedirs(output_dir, exist_ok=True)
    model_files = {}
    tile_names = set()
    for model_name in MULTI_FORECAST_MODELS:
        source_dir = os.path.join(MODEL_WORK_DIR, model_name)
        indexed = {}
        if os.path.isdir(source_dir):
            for filename in os.listdir(source_dir):
                hour_text, tile_filename = filename.split("h_", 1)
                tile = tile_filename.removesuffix(".gtile")
                day = int(hour_text) // 24
                indexed.setdefault((tile, day), []).append(
                    os.path.join(source_dir, filename)
                )
                tile_names.add(tile)
        model_files[model_name] = indexed

    for tile in sorted(tile_names):
        day_entries = []
        for day in range(15):
            components = []
            geometry = None
            for model_name in MULTI_FORECAST_MODELS:
                paths = model_files[model_name].get((tile, day), [])
                daily_high = None
                daily_low = None
                for path in paths:
                    lon_vals, lat_vals, dx, dy, channels = read_binary_tile(path)
                    temperature = channels.get(MODEL_FIELD_CHANNELS["temperature"])
                    if temperature is None:
                        continue
                    geometry = (lon_vals, lat_vals, dx, dy)
                    daily_high = (
                        temperature.copy() if daily_high is None
                        else np.maximum(daily_high, temperature)
                    )
                    daily_low = (
                        temperature.copy() if daily_low is None
                        else np.minimum(daily_low, temperature)
                    )
                if daily_high is not None:
                    components.append((
                        MULTI_FORECAST_HIGH_CHANNELS[model_name],
                        prepare_grid_values(daily_high, 1),
                    ))
                    components.append((
                        MULTI_FORECAST_LOW_CHANNELS[model_name],
                        prepare_grid_values(daily_low, 1),
                    ))
            if geometry is None or not components:
                continue
            lon_vals, lat_vals, dx, dy = geometry
            day_entries.append((
                str(day),
                build_binary_tile_bytes(
                    lon_vals, lat_vals, dx, dy, components
                ),
            ))
        if day_entries:
            write_tile_pack(
                os.path.join(output_dir, f"{tile}.gpack"), day_entries
            )

def load_consensus_skill_weights(path=CONSENSUS_WEIGHTS_FILE):
    try:
        with open(path) as file:
            document = json.load(file)
        return {
            name: max(float(value), 0.0)
            for name, value in document.get("weights", {}).items()
        }
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        return {}

def _family_balanced_weights(model_names, skill_weights=None):
    """Apply learned skill without letting sibling models multiply a family."""
    skill_weights = skill_weights or {}
    family_counts = {}
    for model_name in model_names:
        family = MODEL_FAMILIES[model_name]
        family_counts[family] = family_counts.get(family, 0) + 1
    return np.array([
        max(float(skill_weights.get(model_name, 1.0)), 0.01)
        / family_counts[MODEL_FAMILIES[model_name]]
        for model_name in model_names
    ], dtype=np.float32)

def _weighted_median(stack, weights):
    order = np.argsort(np.where(np.isfinite(stack), stack, np.inf), axis=0)
    sorted_values = np.take_along_axis(stack, order, axis=0)
    broadcast_weights = np.broadcast_to(
        weights.reshape((-1,) + (1,) * (stack.ndim - 1)), stack.shape
    )
    sorted_weights = np.take_along_axis(broadcast_weights, order, axis=0)
    sorted_weights = np.where(np.isfinite(sorted_values), sorted_weights, 0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    threshold = np.sum(sorted_weights, axis=0) * 0.5
    median_index = np.argmax(cumulative >= threshold, axis=0)
    median = np.take_along_axis(
        sorted_values, np.expand_dims(median_index, axis=0), axis=0
    )[0]
    return np.where(threshold > 0, median, np.nan).astype(np.float32)

def _weighted_vote(stack, weights, categories):
    scores = np.stack([
        np.sum(
            np.where(stack == category, weights.reshape((-1, 1, 1)), 0),
            axis=0,
        )
        for category in categories
    ])
    # Reverse argmax resolves exact ties toward the more consequential class.
    reverse_index = np.argmax(scores[::-1], axis=0)
    return np.asarray(categories)[len(categories) - 1 - reverse_index]

def _condition_consensus(stack, weights):
    family_for_code = np.zeros(16, dtype=np.int16)
    for code in (1, 2): family_for_code[code] = 1
    for code in (3, 4, 5): family_for_code[code] = 2
    family_for_code[14] = 3
    for code in (6, 7, 8): family_for_code[code] = 4
    for code in (9, 10, 11, 12): family_for_code[code] = 5
    for code in (13, 15): family_for_code[code] = 6
    rounded = np.clip(np.rint(stack), 0, 15).astype(np.int16)
    condition_families = family_for_code[rounded]
    winning_family = _weighted_vote(
        condition_families, weights, np.arange(7, dtype=np.int16)
    )
    result = np.zeros(stack.shape[1:], dtype=np.float32)
    for family in range(1, 7):
        candidates = np.where(
            condition_families == family, rounded, CONDITION_CODES["Unknown"]
        )
        family_codes = np.where(family_for_code == family)[0]
        selected = _weighted_vote(candidates, weights, family_codes)
        result = np.where(winning_family == family, selected, result)
    return result.astype(np.float32)

def _consensus_spread(stack):
    return (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)).astype(np.float32)

def _confidence_from_spread(field_name, spread, family_count):
    tolerances = {
        "temperature": 8.0,
        "dewpoint": 8.0,
        "wind_u": 12.0,
        "wind_v": 12.0,
        "wind_gust": 15.0,
        "pressure": 2500.0,
        "humidity": 0.5,
        "cloud_cover": 0.7,
        "precipitation": 8.0,
        "snowfall": 5.0,
        "visibility": 15000.0,
        "condition_code": 4.0,
        "precipitation_probability": 1.0,
    }
    agreement = 1.0 - np.clip(spread / tolerances[field_name], 0, 1)
    participation = np.clip(family_count / 3.0, 0, 1)
    return (agreement * participation).astype(np.float32)

def compact_consensus_tiles():
    """Build one robust median/vote forecast from all available models."""
    output_dir = os.path.join(OUTPUT_DIR, "models", "consensus")
    os.makedirs(output_dir, exist_ok=True)
    skill_weights = load_consensus_skill_weights()
    indexed = {}
    for model_name in MODEL_PROVIDER_IDS:
        source_dir = os.path.join(MODEL_WORK_DIR, model_name)
        if not os.path.isdir(source_dir):
            continue
        for filename in os.listdir(source_dir):
            hour_text, tile_filename = filename.split("h_", 1)
            tile = tile_filename.removesuffix(".gtile")
            indexed.setdefault((tile, int(hour_text)), []).append((
                model_name, os.path.join(source_dir, filename)
            ))

    hours = set()
    available_channels = set()
    tile_entries = {}
    for (tile, forecast_hour), paths in indexed.items():
        model_channels = []
        geometry = None
        for model_name, path in paths:
            lon_vals, lat_vals, dx, dy, channels = read_binary_tile(path)
            geometry = (lon_vals, lat_vals, dx, dy)
            model_channels.append((model_name, channels))
        if geometry is None:
            continue

        components = []
        for field_name, channel in MODEL_FIELD_CHANNELS.items():
            if field_name == "precipitation_probability":
                continue
            contributors = [
                (model_name, item[channel])
                for model_name, item in model_channels if channel in item
            ]
            values = [value for _, value in contributors]
            if not values:
                continue
            model_names = [model_name for model_name, _ in contributors]
            family_count = len({MODEL_FAMILIES[name] for name in model_names})
            if family_count < 2:
                continue
            stack = np.stack(values)
            weights = _family_balanced_weights(model_names, skill_weights)
            if field_name == "condition_code":
                consensus = _condition_consensus(stack, weights)
                spread = np.max(stack, axis=0) - np.min(stack, axis=0)
            elif field_name == "precipitation":
                wet = np.where(np.isfinite(stack), stack >= 0.05, False)
                weighted = weights.reshape((-1, 1, 1))
                available_weight = np.sum(
                    np.where(np.isfinite(stack), weighted, 0), axis=0
                )
                wet_probability = np.divide(
                    np.sum(np.where(wet, weighted, 0), axis=0),
                    available_weight,
                    out=np.zeros(stack.shape[1:], dtype=np.float32),
                    where=available_weight > 0,
                )
                wet_stack = np.where(wet, stack, np.nan)
                wet_amount = _weighted_median(wet_stack, weights)
                consensus = np.where(
                    wet_probability >= 0.5,
                    np.nan_to_num(wet_amount, nan=0.0),
                    0.0,
                ).astype(np.float32)
                spread = _consensus_spread(stack)
                probability_channel = MODEL_FIELD_CHANNELS[
                    "precipitation_probability"
                ]
                components.append((
                    probability_channel,
                    prepare_grid_values(
                        wet_probability, 3, preserve_missing=True
                    ),
                ))
                available_channels.add(probability_channel)
            else:
                consensus = _weighted_median(stack, weights)
                spread = _consensus_spread(stack)
            components.append((
                channel,
                prepare_grid_values(consensus, 3, preserve_missing=True),
            ))
            if field_name in CONSENSUS_QUALITY_FIELDS:
                participation = np.full(
                    consensus.shape, family_count, dtype=np.float32
                )
                confidence = _confidence_from_spread(
                    field_name, spread, family_count
                )
                components.extend([
                    (
                        CONSENSUS_CONFIDENCE_CHANNELS[field_name],
                        prepare_grid_values(confidence, 3, preserve_missing=True),
                    ),
                    (
                        CONSENSUS_SPREAD_CHANNELS[field_name],
                        prepare_grid_values(spread, 3, preserve_missing=True),
                    ),
                    (
                        CONSENSUS_PARTICIPATION_CHANNELS[field_name],
                        prepare_grid_values(
                            participation, 0, preserve_missing=True
                        ),
                    ),
                ])
            available_channels.add(channel)
        if not components:
            continue
        lon_vals, lat_vals, dx, dy = geometry
        tile_entries.setdefault(tile, []).append((
            forecast_hour,
            build_binary_tile_bytes(lon_vals, lat_vals, dx, dy, components),
        ))
        hours.add(forecast_hour)

    for tile, entries in tile_entries.items():
        entries.sort(key=lambda item: item[0])
        write_tile_pack(
            os.path.join(output_dir, f"{tile}.gpack"),
            [(str(hour), tile_bytes) for hour, tile_bytes in entries],
        )
    return sorted(hours), available_channels

def compact_model_provider_tiles(reference_time):
    """Pack one location tile's forecast day into one CDN-friendly file."""
    output_root = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(output_root, exist_ok=True)
    manifest = {
        "version": 1,
        "ref_time": reference_time,
        "condition_codes": CONDITION_CODES,
        "field_channels": MODEL_FIELD_CHANNELS,
        "models": {},
    }
    # Build cross-model products before deleting provider intermediates.
    consensus_hours, consensus_channels = compact_consensus_tiles()
    compact_multi_forecast_tiles()
    for model_name in MODEL_PROVIDER_IDS:
        source_dir = os.path.join(MODEL_WORK_DIR, model_name)
        if not os.path.isdir(source_dir):
            continue
        model_dir = os.path.join(output_root, model_name)
        os.makedirs(model_dir, exist_ok=True)
        grouped = {}
        available_channels = set()
        for filename in os.listdir(source_dir):
            hour_text, tile_filename_part = filename.split("h_", 1)
            forecast_hour = int(hour_text)
            tile = tile_filename_part.removesuffix(".gtile")
            day = forecast_hour // 24
            grouped.setdefault((day, tile), []).append(
                (forecast_hour, os.path.join(source_dir, filename))
            )
            with open(os.path.join(source_dir, filename), "rb") as tile_file:
                tile_header = tile_file.read(28)
                channel_count = tile_header[6]
                channel_metadata = tile_file.read(channel_count * 10)
                for index in range(channel_count):
                    available_channels.add(
                        struct.unpack_from("<h", channel_metadata, index * 10)[0]
                    )
        available_hours = set()
        horizon_entries = {}
        for (_, tile), entries in grouped.items():
            available_hours.update(hour for hour, _ in entries)
            horizon_entries.setdefault(tile, []).extend(entries)
        for tile, entries in horizon_entries.items():
            entries.sort(key=lambda item: item[0])
            write_tile_pack(
                os.path.join(model_dir, f"{tile}.gpack"),
                [(str(hour), path) for hour, path in entries],
            )
            for _, path in entries:
                os.remove(path)
        manifest["models"][model_name] = {
            "forecast_hours": sorted(available_hours),
            "day_count": max((hour // 24 for hour in available_hours), default=-1) + 1,
            "pack_layout": "complete_horizon",
            "fields": [
                field_name for field_name, channel in MODEL_FIELD_CHANNELS.items()
                if channel in available_channels
            ],
        }
    if consensus_hours:
        manifest["models"]["consensus"] = {
            "forecast_hours": consensus_hours,
            "day_count": max(hour // 24 for hour in consensus_hours) + 1,
            "pack_layout": "complete_horizon",
            "fields": [
                field_name
                for field_name, channel in MODEL_FIELD_CHANNELS.items()
                if channel in consensus_channels
            ],
            "quality_channels": {
                "confidence": {
                    field: CONSENSUS_CONFIDENCE_CHANNELS[field]
                    for field in CONSENSUS_QUALITY_FIELDS
                },
                "spread": {
                    field: CONSENSUS_SPREAD_CHANNELS[field]
                    for field in CONSENSUS_QUALITY_FIELDS
                },
                "participating_families": {
                    field: CONSENSUS_PARTICIPATION_CHANNELS[field]
                    for field in CONSENSUS_QUALITY_FIELDS
                },
            },
            "model_families": MODEL_FAMILIES,
            "minimum_independent_families": 2,
        }
    with open(os.path.join(output_root, "manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)
    if os.path.isdir(MODEL_WORK_DIR):
        shutil.rmtree(MODEL_WORK_DIR)

def pack_condition_pair(first, second):
    if first is None and second is None:
        return None
    template = first if first is not None else second
    first_values = xr.zeros_like(template) if first is None else first
    second_values = xr.zeros_like(template) if second is None else second
    first_values, second_values = xr.align(first_values, second_values, join="inner")
    return (
        first_values.round().astype(np.int16) & 0x0F
    ) | ((second_values.round().astype(np.int16) & 0x0F) << 4)

def generate_tiles_from_dataset(ds, forecast_hour, job_type, config, min_grid=None, max_grid=None, model_grids=None):
    ref_time_iso = str(ds.time.values)
    print(f"   ✂️ Tiling {job_type}...")

    # --- FOLDER MANAGEMENT ---
    # Create a specific subfolder for this data type (e.g. public/data/wind)
    job_dir = os.path.join(OUTPUT_DIR, job_type)
    os.makedirs(job_dir, exist_ok=True)

    count = 0
    pack_entries = {}
    for lat_start in range(-90, 90, TILE_SIZE):
        for lon_start in range(-180, 180, TILE_SIZE):
            lat_end = min(lat_start + TILE_SIZE, 90)
            lon_end = min(lon_start + TILE_SIZE, 180)

            subset = ds.sel(latitude=slice(lat_end, lat_start), longitude=slice(lon_start, lon_end))
            if subset.latitude.size == 0 or subset.longitude.size == 0: continue

            filename = tile_filename(job_type, forecast_hour, lat_start, lon_start)

            lat_vals = subset.latitude.values
            lon_vals = subset.longitude.values
            dy = (lat_vals[-1] - lat_vals[0]) / (len(lat_vals) - 1) if len(lat_vals) > 1 else 1.0
            dx = (lon_vals[-1] - lon_vals[0]) / (len(lon_vals) - 1) if len(lon_vals) > 1 else 1.0

            subset_min, subset_max = None, None
            if min_grid is not None and max_grid is not None:
                subset_min = min_grid.sel(latitude=slice(lat_end, lat_start), longitude=slice(lon_start, lon_end)).values
                subset_max = max_grid.sel(latitude=slice(lat_end, lat_start), longitude=slice(lon_start, lon_end)).values

            subset_model_grids = {}
            if job_type == "temp":
                subset_model_grids = {
                    model_name: model_grid.sel(
                        latitude=slice(lat_end, lat_start),
                        longitude=slice(lon_start, lon_end)
                    )
                    for model_name, model_grid in (model_grids or {}).items()
                }
            components = make_components(
                subset,
                config,
                job_type,
                subset_min=subset_min,
                subset_max=subset_max,
                model_grids=subset_model_grids
            )
            if not components: continue

            tile_path = os.path.join(job_dir, filename)
            write_binary_tile(tile_path, lon_vals, lat_vals, dx, dy, components)
            pack_start = pack_origin(lat_start, lon_start)
            pack_entries.setdefault(pack_start, []).append(
                (tile_key(lat_start, lon_start), tile_path, lat_start, lon_start)
            )
            count += 1

    pack_count = 0
    for (pack_lat_start, pack_lon_start), pack_tiles in pack_entries.items():
        pack_path = os.path.join(
            job_dir,
            pack_filename(job_type, forecast_hour, pack_lat_start, pack_lon_start)
        )
        entries = [(key, tile_path) for key, tile_path, _, _ in pack_tiles]
        tile_origins = [(lat_start, lon_start) for _, _, lat_start, lon_start in pack_tiles]
        overview_tile = build_pack_overview_tile(
            ds,
            config,
            job_type,
            tile_origins,
            min_grid=min_grid,
            max_grid=max_grid,
            model_grids=model_grids
        )
        if overview_tile:
            entries = entries + [(overview_tile_key(pack_lat_start, pack_lon_start), overview_tile)]
        write_tile_pack(pack_path, entries)
        pack_count += 1

    return ref_time_iso, count, pack_count

def generate_tiles(grib_path, forecast_hour, job_type, config, min_grid=None, max_grid=None, model_grids=None):
    try:
        if job_type == "temp":
            source_datasets = cfgrib.open_datasets(
                grib_path, backend_kwargs={"indexpath": ""}
            )
            temperature = _first_variable(source_datasets, ["t2m", "2t"])
            if temperature is None:
                raise ValueError("2 metre temperature field not found")
            ds = temperature.to_dataset(name="temperature")
        else:
            source_datasets = []
            ds = xr.open_dataset(grib_path, engine='cfgrib')
        ds = normalize_dataset_coordinates(ds)
        ref_time_iso, count, pack_count = generate_tiles_from_dataset(
            ds,
            forecast_hour,
            job_type,
            config,
            min_grid=min_grid,
            max_grid=max_grid,
            model_grids=model_grids
        )
        ds.close()
        for source_ds in source_datasets:
            source_ds.close()
        return True, ref_time_iso, count, pack_count

    except Exception as e:
        print(f"❌ Error tiling {job_type}: {e}")
        return False, None, 0, 0

def gfs_filter_request(date, run_hour, forecast_hour, vars_config, level_config):
    f_str = f"{forecast_hour:03d}"
    server_file = f"gfs.t{run_hour}z.pgrb2.0p25.f{f_str}"
    dir_path = f"/gfs.{date}/{run_hour}/atmos"
    params = {
        'file': server_file,
        'dir': dir_path,
        'leftlon': '0',
        'rightlon': '360',
        'toplat': '90',
        'bottomlat': '-90',
        **vars_config,
        **level_config
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl", params

def gfs_atmos_base_url(date, run_hour, forecast_hour):
    f_str = f"{forecast_hour:03d}"
    return (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
        f"gfs.{date}/{run_hour}/atmos/gfs.t{run_hour}z.pgrb2.0p25.f{f_str}"
    )

def download_grib(url, params, output_path):
    response = requests.get(url, params=params, stream=True, timeout=180)
    if response.status_code != 200:
        print(f"⚠️ NOAA Error {response.status_code}")
        return False

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=16384):
            f.write(chunk)
    return True

def download_ecmwf_grib(model, date_str, run_hour, forecast_hour, output_path):
    """
    Downloads ECMWF IFS or AIFS GRIB2 file from AWS S3 public bucket.
    Tries the matching date/run_hour first, then falls back to preceding runs.
    """
    base_url = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
    dt = datetime.datetime.strptime(date_str + run_hour, "%Y%m%d%H")
    
    # Map model key to correct S3 folder name
    s3_model = "aifs-single" if model == "aifs" else model
    
    for i in range(5):
        try_dt = dt - datetime.timedelta(hours=6 * i)
        try_date = try_dt.strftime("%Y%m%d")
        try_run = f"{try_dt.hour:02d}"
        adjusted_step = forecast_hour + (6 * i)
        
        filename = f"{try_date}{try_run}0000-{adjusted_step}h-oper-fc.grib2"
        url = f"{base_url}/{try_date}/{try_run}z/{s3_model}/0p25/oper/{filename}"
        
        print(f"      Trying ECMWF {model} URL: {url}")
        res = requests.get(url, stream=True, timeout=30)
        if res.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=65536):
                    f.write(chunk)
            print(f"      ✅ Successfully downloaded ECMWF {model} from run {try_date} {try_run}z (step +{adjusted_step}h)")
            return True
        elif res.status_code == 404:
            continue
        else:
            print(f"      ⚠️ ECMWF HTTP status {res.status_code} for {url}")
            
    print(f"      ❌ Failed to download ECMWF {model} for GFS run {date_str} {run_hour}z, step +{forecast_hour}h")
    return False

def download_aigfs_grib(date_str, run_hour, forecast_hour, output_path):
    """
    Downloads NOAA AIGFS GRIB2 file from NOMADS.
    Tries the matching date/run_hour first, then falls back to preceding runs.
    """
    base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod"
    dt = datetime.datetime.strptime(date_str + run_hour, "%Y%m%d%H")
    
    for i in range(5):
        try_dt = dt - datetime.timedelta(hours=6 * i)
        try_date = try_dt.strftime("%Y%m%d")
        try_run = f"{try_dt.hour:02d}"
        adjusted_step = forecast_hour + (6 * i)
        
        # AIGFS uses 'sfc' product identifier for surface forecasts
        url = f"{base_url}/aigfs.{try_date}/{try_run}/model/atmos/grib2/aigfs.t{try_run}z.sfc.f{adjusted_step:03d}.grib2"
        
        print(f"      Trying NOAA AIGFS URL: {url}")
        res = requests.get(url, stream=True, timeout=30)
        if res.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=65536):
                    f.write(chunk)
            print(f"      ✅ Successfully downloaded NOAA AIGFS from run {try_date} {try_run}z (step +{adjusted_step}h)")
            return True
        elif res.status_code == 404:
            continue
        else:
            print(f"      ⚠️ NOAA AIGFS HTTP status {res.status_code}")
            
    print(f"      ❌ Failed to download NOAA AIGFS for GFS run {date_str} {run_hour}z, step +{forecast_hour}h")
    return False

def download_icon_parameter(
    date_str,
    run_hour,
    forecast_hour,
    parameter,
    file_parameter,
    output_path,
):
    """Download one compact ICON-EU regular-lat-lon parameter."""
    base_url = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
    bz2_path = output_path + ".bz2"
    dt = datetime.datetime.strptime(date_str + run_hour, "%Y%m%d%H")
    
    for i in range(5):
        try_dt = dt - datetime.timedelta(hours=6 * i)
        try_date_str = try_dt.strftime("%Y%m%d%H")
        try_run = f"{try_dt.hour:02d}"
        adjusted_step = forecast_hour + (6 * i)
        
        url = f"{base_url}/{try_run}/{parameter}/icon-eu_europe_regular-lat-lon_single-level_{try_date_str}_{adjusted_step:03d}_{file_parameter}.grib2.bz2"
        
        print(f"      Trying DWD ICON-EU URL: {url}")
        res = requests.get(url, stream=True, timeout=30)
        if res.status_code == 200:
            with open(bz2_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=65536):
                    f.write(chunk)
            
            # Decompress bz2
            import bz2
            try:
                with bz2.open(bz2_path, "rb") as source, open(output_path, "wb") as dest:
                    dest.write(source.read())
                if os.path.exists(bz2_path):
                    os.remove(bz2_path)
                print(f"      ✅ Downloaded ICON-EU {parameter} from {try_date_str}z (+{adjusted_step}h)")
                return True
            except Exception as decompress_err:
                print(f"      ⚠️ Decompress error for DWD ICON-EU: {decompress_err}")
                if os.path.exists(bz2_path):
                    os.remove(bz2_path)
                continue
        elif res.status_code == 404:
            continue
        else:
            print(f"      ⚠️ DWD ICON-EU HTTP status {res.status_code}")
            
    print(f"      ❌ Failed ICON-EU {parameter} for {date_str} {run_hour}z +{forecast_hour}h")
    return False

def download_icon_grib(date_str, run_hour, forecast_hour, output_path):
    return download_icon_parameter(
        date_str,
        run_hour,
        forecast_hour,
        "t_2m",
        "T_2M",
        output_path,
    )

def load_single_model_field(path, target):
    datasets = []
    try:
        datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
        field = first_data_array(datasets[0]) if datasets else None
        return _aligned_field(field, target)
    finally:
        for ds in datasets:
            ds.close()

def icon_weather_code_grid(weather_code, cloud_cover, target):
    """Map ICON's WMO present-weather number to Gusty condition IDs."""
    if cloud_cover is None:
        result = xr.full_like(target, CONDITION_CODES["Unknown"], dtype=np.float32)
    else:
        result = derive_condition_grid(target, cloud_cover=cloud_cover)
    if weather_code is None:
        return result
    code = weather_code.round()
    result = xr.where((code >= 45) & (code <= 48), CONDITION_CODES["Foggy"], result)
    result = xr.where((code >= 51) & (code <= 57), CONDITION_CODES["Drizzle"], result)
    result = xr.where(((code >= 61) & (code <= 65)) | ((code >= 80) & (code <= 82)), CONDITION_CODES["Rain"], result)
    result = xr.where((code == 66) | (code == 67), CONDITION_CODES["FreezingRain"], result)
    result = xr.where(((code >= 68) & (code <= 79)) | ((code >= 85) & (code <= 86)), CONDITION_CODES["Snow"], result)
    result = xr.where((code >= 95) & (code <= 99), CONDITION_CODES["Thunderstorms"], result)
    result = xr.where((code == 96) | (code == 99), CONDITION_CODES["Hail"], result)
    return result.astype(np.float32)

def process_multi_model_variance(date_str, run_hour, forecast_hour, gfs_grib_path):
    """
    Downloads other models (ECMWF IFS, ECMWF AIFS, NOAA AIGFS, DWD ICON-EU)
    for the given date/run/step, opens them, aligns/regrids them to the GFS grid,
    and returns min/max temperature DataArrays.
    """
    print(f"   🔍 Computing temperature variance from other models...")
    
    # Open GFS reference
    try:
        gfs_source_datasets = cfgrib.open_datasets(
            gfs_grib_path, backend_kwargs={"indexpath": ""}
        )
        gfs_temp = _first_variable(gfs_source_datasets, ["t2m", "2t"])
        if gfs_temp is None:
            raise ValueError("GFS 2 metre temperature field not found")
        gfs_ds = gfs_temp.to_dataset(name="temperature")
        gfs_ds = normalize_dataset_coordinates(gfs_ds)
        gfs_temp = first_data_array(gfs_ds)  # Keep in Kelvin
    except Exception as e:
        print(f"      ⚠️ GFS open error: {e}")
        return None, None, None, None
        
    models_temps = {"gfs": gfs_temp}
    provider_fields = {
        "gfs": extract_provider_fields("gfs", gfs_grib_path, gfs_temp, gfs_temp)
    }
    
    # Output paths
    ifs_path = os.path.join(OUTPUT_DIR, f"temp_variance_ifs_{forecast_hour}.grib2")
    aifs_path = os.path.join(OUTPUT_DIR, f"temp_variance_aifs_{forecast_hour}.grib2")
    aigfs_path = os.path.join(OUTPUT_DIR, f"temp_variance_aigfs_{forecast_hour}.grib2")
    icon_path = os.path.join(OUTPUT_DIR, f"temp_variance_icon_{forecast_hour}.grib2")

    download_jobs = {
        "ifs": lambda: download_ecmwf_grib(
            "ifs", date_str, run_hour, forecast_hour, ifs_path
        ),
        "aifs": lambda: download_ecmwf_grib(
            "aifs", date_str, run_hour, forecast_hour, aifs_path
        ),
        "aigfs": lambda: download_aigfs_grib(
            date_str, run_hour, forecast_hour, aigfs_path
        ),
    }
    if forecast_hour <= 120:
        download_jobs["icon"] = lambda: download_icon_grib(
            date_str, run_hour, forecast_hour, icon_path
        )
    downloads = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(download_jobs)
    ) as executor:
        futures = {
            name: executor.submit(download) for name, download in download_jobs.items()
        }
        for name, future in futures.items():
            try:
                downloads[name] = future.result()
            except Exception as error:
                print(f"      ⚠️ {name} download failed: {error}")
                downloads[name] = False
    
    # ECMWF IFS (use filter_by_keys to avoid height conflicts)
    if downloads.get("ifs"):
        try:
            ds = xr.open_dataset(ifs_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2.0})
            ds = normalize_dataset_coordinates(ds)
            temp = first_data_array(ds)  # Keep in Kelvin
            temp_aligned = temp.interp_like(gfs_temp, method='linear')
            models_temps["ifs"] = temp_aligned.load()
            provider_fields["ifs"] = extract_provider_fields(
                "ifs", ifs_path, gfs_temp, models_temps["ifs"]
            )
            ds.close()
        except Exception as e:
            print(f"      ⚠️ Error processing ECMWF IFS: {e}")
        finally:
            if os.path.exists(ifs_path): os.remove(ifs_path)
            
    # ECMWF AIFS (use filter_by_keys to avoid height conflicts)
    if downloads.get("aifs"):
        try:
            ds = xr.open_dataset(aifs_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2.0})
            ds = normalize_dataset_coordinates(ds)
            temp = first_data_array(ds)  # Keep in Kelvin
            temp_aligned = temp.interp_like(gfs_temp, method='linear')
            models_temps["aifs"] = temp_aligned.load()
            provider_fields["aifs"] = extract_provider_fields(
                "aifs", aifs_path, gfs_temp, models_temps["aifs"]
            )
            ds.close()
        except Exception as e:
            print(f"      ⚠️ Error processing ECMWF AIFS: {e}")
        finally:
            if os.path.exists(aifs_path): os.remove(aifs_path)
            
    # NOAA AIGFS (use filter_by_keys just in case)
    if downloads.get("aigfs"):
        try:
            ds = xr.open_dataset(aigfs_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2.0})
            ds = normalize_dataset_coordinates(ds)
            temp = first_data_array(ds)  # Keep in Kelvin
            temp_aligned = temp.interp_like(gfs_temp, method='linear')
            models_temps["aigfs"] = temp_aligned.load()
            provider_fields["aigfs"] = extract_provider_fields(
                "aigfs", aigfs_path, gfs_temp, models_temps["aigfs"]
            )
            ds.close()
        except Exception as e:
            print(f"      ⚠️ Error processing NOAA AIGFS: {e}")
        finally:
            if os.path.exists(aigfs_path): os.remove(aigfs_path)
            
    # DWD ICON-EU ends at 120 hours; the global models continue to day 14.
    if downloads.get("icon"):
        try:
            ds = xr.open_dataset(icon_path, engine='cfgrib')
            ds = normalize_dataset_coordinates(ds)
            temp = first_data_array(ds)  # Keep in Kelvin
            temp_aligned = temp.interp_like(gfs_temp, method='linear')
            models_temps["icon"] = temp_aligned.load()
            provider_fields["icon"] = extract_provider_fields(
                "icon", icon_path, gfs_temp, models_temps["icon"]
            )
            icon_supplements = {
                "cloud_cover": ("clct", "CLCT"),
                "precipitation": ("tot_prec", "TOT_PREC"),
                "snowfall": ("snow_gsp", "SNOW_GSP"),
                "weather_code": ("ww", "WW"),
            }
            loaded_supplements = {}
            for field_name, (parameter, file_parameter) in icon_supplements.items():
                supplement_path = os.path.join(
                    OUTPUT_DIR,
                    f"temp_provider_icon_{parameter}_{forecast_hour}.grib2",
                )
                try:
                    if download_icon_parameter(
                        date_str,
                        run_hour,
                        forecast_hour,
                        parameter,
                        file_parameter,
                        supplement_path,
                    ):
                        loaded_supplements[field_name] = load_single_model_field(
                            supplement_path, gfs_temp
                        )
                finally:
                    if os.path.exists(supplement_path):
                        os.remove(supplement_path)

            icon_fields = provider_fields["icon"]
            cloud = loaded_supplements.get("cloud_cover")
            if cloud is not None:
                icon_fields["cloud_cover"] = xr.where(
                    cloud > 1.5, cloud / 100.0, cloud
                ).clip(0, 1)
            for field_name in ("precipitation", "snowfall"):
                field = loaded_supplements.get(field_name)
                if field is not None:
                    field = _interval_accumulation(
                        "icon", field_name, field, field.attrs
                    )
                    normalized = _precipitation_mm(field)
                    normalized.attrs["units"] = "mm"
                    icon_fields[field_name] = normalized
            icon_fields["condition_code"] = icon_weather_code_grid(
                loaded_supplements.get("weather_code"),
                cloud,
                gfs_temp,
            )
            ds.close()
        except Exception as e:
            print(f"      ⚠️ Error processing DWD ICON-EU: {e}")
        finally:
            if os.path.exists(icon_path): os.remove(icon_path)

    gfs_ds.close()
    for source_ds in gfs_source_datasets:
        source_ds.close()
    
    # Compute min/max
    if len(models_temps) > 1:
        model_names = list(models_temps.keys())
        aligned_models = xr.align(*models_temps.values(), join='inner')
        aligned_model_grids = dict(zip(model_names, aligned_models))
        combined = xr.concat(aligned_models, dim='model')
        min_grid = combined.min(dim='model')
        max_grid = combined.max(dim='model')
        print(f"      ✅ Computed variance from {len(models_temps)} models.")
        for model_name, fields in provider_fields.items():
            write_model_provider_hour(model_name, forecast_hour, fields)
        return min_grid, max_grid, aligned_model_grids, provider_fields
    else:
        print("      ⚠️ No other models downloaded. Using GFS values for min/max.")
        write_model_provider_hour(
            "gfs", forecast_hour, provider_fields["gfs"]
        )
        return gfs_temp, gfs_temp, {"gfs": gfs_temp}, provider_fields

def first_data_array(ds):
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError("No data variables found")
    return ds[data_vars[0]]

def storm_potential_index(
    cape,
    composite_reflectivity,
    cin=None,
    precipitation_rate=None,
    convective_precipitation_rate=None,
    precipitable_water=None,
    dewpoint=None,
    wind_u_500=None,
    wind_v_500=None,
    wind_u_850=None,
    wind_v_850=None
):
    cape_score = ((cape - 250.0) / 1750.0).clip(0.0, 1.0)
    reflectivity_score = ((composite_reflectivity - 18.0) / 37.0).clip(0.0, 1.0)
    precip_scores = []

    if precipitation_rate is not None:
        # GFS PRATE/CPRAT is kg m-2 s-1, which is equivalent to mm/s for water.
        precip_scores.append(((precipitation_rate * 3600.0 - 0.25) / 8.0).clip(0.0, 1.0))
    if convective_precipitation_rate is not None:
        precip_scores.append(((convective_precipitation_rate * 3600.0 - 0.10) / 5.0).clip(0.0, 1.0))

    convective_signal = reflectivity_score
    for score in precip_scores:
        convective_signal = xr.where(convective_signal >= score, convective_signal, score)

    # CAPE identifies unstable air, while composite reflectivity anchors the signal
    # to model-predicted convection. The overlap term keeps broad unstable air from
    # becoming a blanket high-risk layer by itself.
    overlap_score = np.sqrt(cape_score) * np.power(convective_signal, 0.7)
    index = (0.25 * cape_score) + (0.55 * overlap_score) + (0.20 * convective_signal * cape_score)

    moisture_scores = []
    if precipitable_water is not None:
        moisture_scores.append(((precipitable_water - 20.0) / 25.0).clip(0.0, 1.0))
    if dewpoint is not None:
        dewpoint_c = xr.where(dewpoint > 150.0, dewpoint - 273.15, dewpoint)
        moisture_scores.append(((dewpoint_c - 12.0) / 12.0).clip(0.0, 1.0))

    if moisture_scores:
        moisture_score = moisture_scores[0]
        for score in moisture_scores[1:]:
            moisture_score = xr.where(moisture_score >= score, moisture_score, score)
        index = index * (0.55 + (0.45 * moisture_score))

    if cin is not None:
        cin_abs = np.abs(cin)
        cap_release_score = (1.0 - ((cin_abs - 25.0) / 175.0).clip(0.0, 1.0)).clip(0.0, 1.0)
        index = index * (0.35 + (0.65 * cap_release_score))

    if all(item is not None for item in [wind_u_500, wind_v_500, wind_u_850, wind_v_850]):
        shear = np.sqrt(
            np.square(wind_u_500 - wind_u_850) +
            np.square(wind_v_500 - wind_v_850)
        )
        shear_score = ((shear - 8.0) / 18.0).clip(0.0, 1.0)
        index = index * (0.85 + (0.30 * shear_score))

    return index.clip(0.0, 1.0)

def open_optional_storm_sources(source_paths, source_datasets, source_names):
    optional_arrays = {}
    for source_name in source_names:
        path = source_paths.get(source_name)
        if not path:
            continue
        ds = normalize_dataset_coordinates(xr.open_dataset(path, engine='cfgrib'))
        source_datasets.append(ds)
        optional_arrays[source_name] = first_data_array(ds).astype(np.float32)
    return optional_arrays

def download_and_process_storm_potential(job_type, date, run_hour, forecast_hour, config):
    source_paths = {}
    source_datasets = []
    f_str = f"{forecast_hour:03d}"

    try:
        base_url = gfs_atmos_base_url(date, run_hour, forecast_hour)
        for source_name, source_config in config["sources"].items():
            path = os.path.join(OUTPUT_DIR, f"temp_{job_type}_{source_name}_{forecast_hour}.grib2")
            print(f"⬇️ {job_type.upper()} {source_name} f{f_str}...")
            if not download_idx_message(base_url, source_config["idx_match"], path):
                if source_config.get("optional", False):
                    print(f"   ↪️ Optional storm source skipped: {source_name}")
                    continue
                return None, 0, 0
            source_paths[source_name] = path

        cape_ds = normalize_dataset_coordinates(xr.open_dataset(source_paths["cape"], engine='cfgrib'))
        refc_ds = normalize_dataset_coordinates(xr.open_dataset(source_paths["composite_reflectivity"], engine='cfgrib'))
        source_datasets.extend([cape_ds, refc_ds])

        optional_source_names = [
            "cin",
            "precipitation_rate",
            "convective_precipitation_rate",
            "precipitable_water",
            "dewpoint",
            "wind_u_500",
            "wind_v_500",
            "wind_u_850",
            "wind_v_850"
        ]
        optional_arrays = open_optional_storm_sources(
            source_paths,
            source_datasets,
            optional_source_names
        )

        align_names = list(optional_arrays.keys())
        aligned = xr.align(
            first_data_array(cape_ds).astype(np.float32),
            first_data_array(refc_ds).astype(np.float32),
            *[optional_arrays[name] for name in align_names],
            join="inner"
        )
        cape = aligned[0]
        refc = aligned[1]
        optional_arrays = {
            name: aligned[index + 2]
            for index, name in enumerate(align_names)
        }
        storm_score = storm_potential_index(
            cape,
            refc,
            cin=optional_arrays.get("cin"),
            precipitation_rate=optional_arrays.get("precipitation_rate"),
            convective_precipitation_rate=optional_arrays.get("convective_precipitation_rate"),
            precipitable_water=optional_arrays.get("precipitable_water"),
            dewpoint=optional_arrays.get("dewpoint"),
            wind_u_500=optional_arrays.get("wind_u_500"),
            wind_v_500=optional_arrays.get("wind_v_500"),
            wind_u_850=optional_arrays.get("wind_u_850"),
            wind_v_850=optional_arrays.get("wind_v_850")
        )
        if config["grid_type"] == "vector":
            wind_u_ds = normalize_dataset_coordinates(xr.open_dataset(source_paths["wind_u"], engine='cfgrib'))
            wind_v_ds = normalize_dataset_coordinates(xr.open_dataset(source_paths["wind_v"], engine='cfgrib'))
            source_datasets.extend([wind_u_ds, wind_v_ds])

            storm_score, wind_u, wind_v = xr.align(
                storm_score,
                first_data_array(wind_u_ds),
                first_data_array(wind_v_ds),
                join="inner"
            )
            storm_ds = xr.Dataset({
                "u_storm_motion": wind_u.astype(np.float32) * storm_score,
                "v_storm_motion": wind_v.astype(np.float32) * storm_score
            })
        else:
            storm_ds = (storm_score * 100.0).to_dataset(name="storm_potential")

        ref_time, count, pack_count = generate_tiles_from_dataset(storm_ds, forecast_hour, job_type, config)
        return ref_time, count, pack_count
    except Exception as e:
        print(f"❌ Error generating {job_type}: {e}")
        return None, 0, 0
    finally:
        for ds in source_datasets:
            ds.close()
        for path in source_paths.values():
            if os.path.exists(path):
                os.remove(path)

def download_and_process(job_type, date, run_hour, forecast_hour):
    if job_type not in NOAA_CONFIG: return None, 0, 0
    conf = NOAA_CONFIG[job_type]
    f_str = f"{forecast_hour:03d}"
    
    url = ""
    params = {}
    
    if conf['type'] == 'direct_download':
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date}/{run_hour}/wave/gridded/gfswave.t{run_hour}z.global.0p25.f{f_str}.grib2"
    elif conf['type'] == 'derived_storm_potential':
        return download_and_process_storm_potential(job_type, date, run_hour, forecast_hour, conf)
    elif conf['type'] == 'gefs_chem_idx':
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{date}/{run_hour}/chem/pgrb2ap25/gefs.chem.t{run_hour}z.a2d_0p25.f{f_str}.grib2"
    elif conf['type'] == 'gefs_aero':
        url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_aer_0p50.pl"
        server_file = f"gec00.t{run_hour}z.pgrb2a.0p50.f{f_str}"
        dir_path = f"/gefs.{date}/{run_hour}/chem/pgrb2a.0p50"
        params = {'file': server_file, 'dir': dir_path, 'leftlon': '0', 'rightlon': '360', 'toplat': '90', 'bottomlat': '-90', **conf['vars'], **conf['level']}
    else:
        url, params = gfs_filter_request(date, run_hour, forecast_hour, conf['vars'], conf['level'])

    grib_path = os.path.join(OUTPUT_DIR, f"temp_{job_type}_{forecast_hour}.grib2")

    try:
        print(f"⬇️ {job_type.upper()} f{f_str}...")
        if conf['type'] == 'gefs_chem_idx':
            downloaded = download_idx_message(url, conf["idx_match"], grib_path)
            if not downloaded:
                return None, 0, 0
            success, ref_time, count, pack_count = generate_tiles(grib_path, forecast_hour, job_type, conf)
            if os.path.exists(grib_path): os.remove(grib_path)
            return ref_time, count, pack_count

        response = requests.get(url, params=params, stream=True, timeout=180)
        if response.status_code == 200:
            with open(grib_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    f.write(chunk)
            
            min_grid, max_grid, model_grids = None, None, None
            if job_type == "temp":
                min_grid, max_grid, model_grids, _ = process_multi_model_variance(
                    date, run_hour, forecast_hour, grib_path
                )

            success, ref_time, count, pack_count = generate_tiles(
                grib_path,
                forecast_hour,
                job_type,
                conf,
                min_grid=min_grid,
                max_grid=max_grid,
                model_grids=model_grids
            )
            if os.path.exists(grib_path): os.remove(grib_path)
            return ref_time, count, pack_count
        else:
            print(f"⚠️ NOAA Error {response.status_code}")
            return None, 0, 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None, 0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="Data type to fetch", default="all")
    args = parser.parse_args()

    if args.type == "all":
        print("--- Cleaning Output Directory ---")
        clean_output_directory()

    date, run = get_latest_run_time()
    print(f"--- Starting Fetch: {date} {run}z ---")

    jobs = list(NOAA_CONFIG.keys()) if args.type == "all" else [args.type]
    manifest_updates = {}
    
    for job in jobs:
        print(f"\n--- Processing: {job} ---")
        job_ref_time = None
        total_tiles = 0
        total_packs = 0
        forecast_hours = (
            TEMPERATURE_HOURS_TO_FETCH if job == "temp" else HOURS_TO_FETCH
        )
        for hour in forecast_hours:
            ref_time, count, pack_count = download_and_process(job, date, run, hour)
            if ref_time: job_ref_time = ref_time
            total_tiles += count
            total_packs += pack_count
            time.sleep(1)
        
        if job_ref_time:
            manifest_entry = {
                "ref_time": job_ref_time,
                "tiles": total_tiles,
                "packs": total_packs,
                "grid_type": NOAA_CONFIG[job]["grid_type"],
                "tile_format": TILE_FORMAT,
                "pack_format": PACK_FORMAT
            }
            if "metadata" in NOAA_CONFIG[job]:
                manifest_entry["metadata"] = NOAA_CONFIG[job]["metadata"]
            manifest_updates[job] = manifest_entry
            if job == "temp":
                compact_model_provider_tiles(job_ref_time)

    if manifest_updates:
        print("\n✅ Batch Complete.")
        with open(os.path.join(OUTPUT_DIR, "manifest_update.json"), 'w') as f:
            json.dump(manifest_updates, f, indent=2)
