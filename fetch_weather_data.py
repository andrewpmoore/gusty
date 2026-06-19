import os
import shutil
import requests
import datetime
import time
import json
import argparse
import sys
import struct
import math
import numpy as np
import xarray as xr

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
        "vars": {"var_TMP": "on"},
        "level": {"lev_2_m_above_ground": "on"},
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

def prepare_grid_values(values, precision):
    filled = np.where(np.isnan(values), 0, values)
    return np.round(filled.astype(np.float32), precision).flatten()

def quantize_int16(values):
    min_value = float(np.min(values))
    max_value = float(np.max(values))

    if min_value == max_value:
        return np.zeros(values.shape, dtype="<i2"), 1.0, min_value

    scale = (max_value - min_value) / (INT16_MAX_VALUE - INT16_MIN_VALUE)
    offset = (max_value + min_value) / 2.0
    quantized = np.rint((values - offset) / scale)
    quantized = np.clip(quantized, INT16_MIN_VALUE, INT16_MAX_VALUE).astype("<i2")
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

def make_components(subset, config, job_type, sample_step=1):
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

    return components

def build_pack_overview_tile(ds, config, job_type, tile_origins):
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
    components = make_components(subset, config, job_type, OVERVIEW_SAMPLE_STEP)
    if not components:
        return None

    return build_binary_tile_bytes(lon_vals, lat_vals, dx, dy, components)

def normalize_dataset_coordinates(ds):
    return ds.assign_coords(
        longitude=(((ds.longitude + 180) % 360) - 180)
    ).sortby('longitude')

def generate_tiles_from_dataset(ds, forecast_hour, job_type, config):
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

            components = make_components(subset, config, job_type)
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
        overview_tile = build_pack_overview_tile(ds, config, job_type, tile_origins)
        if overview_tile:
            entries = entries + [(overview_tile_key(pack_lat_start, pack_lon_start), overview_tile)]
        write_tile_pack(pack_path, entries)
        pack_count += 1

    return ref_time_iso, count, pack_count

def generate_tiles(grib_path, forecast_hour, job_type, config):
    try:
        ds = xr.open_dataset(grib_path, engine='cfgrib')
        ds = normalize_dataset_coordinates(ds)
        ref_time_iso, count, pack_count = generate_tiles_from_dataset(ds, forecast_hour, job_type, config)
        ds.close()
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
            success, ref_time, count, pack_count = generate_tiles(grib_path, forecast_hour, job_type, conf)
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
        for hour in HOURS_TO_FETCH:
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

    if manifest_updates:
        print("\n✅ Batch Complete.")
        with open(os.path.join(OUTPUT_DIR, "manifest_update.json"), 'w') as f:
            json.dump(manifest_updates, f, indent=2)
