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
        "type": "gefs_aero", 
        "vars": {"var_PMTF": "on"}, 
        "level": {"lev_surface": "on"},
        "grid_type": "scalar"
    },
    "dust": {
        "type": "gefs_aero",
        "vars": {"var_DUST": "on"}, 
        "level": {"lev_surface": "on"},
        "grid_type": "scalar"
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

def precision_for_job(job_type):
    return 3 if job_type in ["snow", "smoke", "dust"] else 1

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

def write_binary_tile(path, lon_vals, lat_vals, dx, dy, components):
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

    with open(path, "wb") as f:
        f.write(header)
        for quantized in quantized_components:
            f.write(quantized.tobytes())

def write_tile_pack(pack_path, entries):
    header = bytearray()
    data = bytearray()
    table = []

    for key, tile_path in entries:
        with open(tile_path, "rb") as f:
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

def generate_tiles(grib_path, forecast_hour, job_type, config):
    try:
        ds = xr.open_dataset(grib_path, engine='cfgrib')
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
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

                components = []
                if config['grid_type'] == 'vector':
                    data_vars = list(subset.data_vars)
                    if len(data_vars) < 2: continue
                    var1, var2 = subset[data_vars[0]], subset[data_vars[1]]
                    components.append((2, prepare_grid_values(var1.values, 1)))
                    components.append((3, prepare_grid_values(var2.values, 1)))
                else:
                    data_vars = list(subset.data_vars)
                    if not data_vars: continue
                    var1 = subset[data_vars[0]]
                    components.append((0, prepare_grid_values(var1.values, precision_for_job(job_type))))

                tile_path = os.path.join(job_dir, filename)
                write_binary_tile(tile_path, lon_vals, lat_vals, dx, dy, components)
                pack_start = pack_origin(lat_start, lon_start)
                pack_entries.setdefault(pack_start, []).append((tile_key(lat_start, lon_start), tile_path))
                count += 1

        pack_count = 0
        for (pack_lat_start, pack_lon_start), entries in pack_entries.items():
            pack_path = os.path.join(
                job_dir,
                pack_filename(job_type, forecast_hour, pack_lat_start, pack_lon_start)
            )
            write_tile_pack(pack_path, entries)
            pack_count += 1

        ds.close()
        return True, ref_time_iso, count, pack_count

    except Exception as e:
        print(f"❌ Error tiling {job_type}: {e}")
        return False, None, 0, 0

def download_and_process(job_type, date, run_hour, forecast_hour):
    if job_type not in NOAA_CONFIG: return None, 0
    conf = NOAA_CONFIG[job_type]
    f_str = f"{forecast_hour:03d}"
    
    url = ""
    params = {}
    
    if conf['type'] == 'direct_download':
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date}/{run_hour}/wave/gridded/gfswave.t{run_hour}z.global.0p25.f{f_str}.grib2"
    elif conf['type'] == 'gefs_aero':
        url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_aer_0p50.pl"
        server_file = f"gec00.t{run_hour}z.pgrb2a.0p50.f{f_str}"
        dir_path = f"/gefs.{date}/{run_hour}/chem/pgrb2a.0p50"
        params = {'file': server_file, 'dir': dir_path, 'leftlon': '0', 'rightlon': '360', 'toplat': '90', 'bottomlat': '-90', **conf['vars'], **conf['level']}
    else:
        url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        server_file = f"gfs.t{run_hour}z.pgrb2.0p25.f{f_str}"
        dir_path = f"/gfs.{date}/{run_hour}/atmos"
        params = {'file': server_file, 'dir': dir_path, 'leftlon': '0', 'rightlon': '360', 'toplat': '90', 'bottomlat': '-90', **conf['vars'], **conf['level']}

    grib_path = os.path.join(OUTPUT_DIR, f"temp_{job_type}_{forecast_hour}.grib2")

    try:
        print(f"⬇️ {job_type.upper()} f{f_str}...")
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
            manifest_updates[job] = {
                "ref_time": job_ref_time,
                "tiles": total_tiles,
                "packs": total_packs,
                "grid_type": NOAA_CONFIG[job]["grid_type"],
                "tile_format": TILE_FORMAT,
                "pack_format": PACK_FORMAT
            }

    if manifest_updates:
        print("\n✅ Batch Complete.")
        with open(os.path.join(OUTPUT_DIR, "manifest_update.json"), 'w') as f:
            json.dump(manifest_updates, f, indent=2)
