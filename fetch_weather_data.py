import os
import shutil
import requests
import datetime
import time
import json
import argparse
import sys
import numpy as np
import xarray as xr

# --- CONFIGURATION REGISTRY ---
NOAA_CONFIG = {
    "wind": {
        "type": "gfs_atmos",
        "vars": {"var_UGRD": "on", "var_VGRD": "on"},
        "level": {"lev_10_m_above_ground": "on"},
        "json_type": "vector"
    },
    "temp": {
        "type": "gfs_atmos",
        "vars": {"var_TMP": "on"},
        "level": {"lev_2_m_above_ground": "on"},
        "json_type": "scalar"
    },
    "humidity": {
        "type": "gfs_atmos",
        "vars": {"var_RH": "on"},
        "level": {"lev_2_m_above_ground": "on"},
        "json_type": "scalar"
    },
    "pressure": {
        "type": "gfs_atmos",
        "vars": {"var_PRMSL": "on"},
        "level": {"lev_mean_sea_level": "on"},
        "json_type": "scalar"
    },
    "waves": {
        "type": "direct_download", 
        "vars": {}, 
        "level": {},
        "json_type": "vector"
    },
    "smoke": {
        "type": "gefs_aero", 
        "vars": {"var_PMTF": "on"}, 
        "level": {"lev_surface": "on"},
        "json_type": "scalar"
    },
    "dust": {
        "type": "gefs_aero",
        "vars": {"var_DUST": "on"}, 
        "level": {"lev_surface": "on"},
        "json_type": "scalar"
    },
    "snow": {
        "type": "gfs_atmos",
        "vars": {"var_SNOD": "on"},
        "level": {"lev_surface": "on"},
        "json_type": "scalar"
    },
    "cape": {
        "type": "gfs_atmos",
        "vars": {"var_CAPE": "on"},
        "level": {"lev_surface": "on"},
        "json_type": "scalar"
    }
}

OUTPUT_DIR = "public/data"
HOURS_TO_FETCH = [0, 3, 6, 9, 12, 15, 18, 21, 24, 36, 48, 72]
TILE_SIZE = 20

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
        for lat_start in range(-90, 90, TILE_SIZE):
            for lon_start in range(-180, 180, TILE_SIZE):
                
                lat_end = min(lat_start + TILE_SIZE, 90)
                lon_end = min(lon_start + TILE_SIZE, 180)

                subset = ds.sel(latitude=slice(lat_end, lat_start), longitude=slice(lon_start, lon_end))
                if subset.latitude.size == 0 or subset.longitude.size == 0: continue

                # Filename scheme
                lat_label = f"N{abs(lat_start)}" if lat_start >= 0 else f"S{abs(lat_start)}"
                lon_label = f"E{abs(lon_start)}" if lon_start >= 0 else f"W{abs(lon_start)}"
                filename = f"{job_type}_{forecast_hour}h_{lat_label}_{lon_label}.json"

                lat_vals = subset.latitude.values
                lon_vals = subset.longitude.values
                dy = (lat_vals[-1] - lat_vals[0]) / (len(lat_vals) - 1) if len(lat_vals) > 1 else 1.0
                dx = (lon_vals[-1] - lon_vals[0]) / (len(lon_vals) - 1) if len(lon_vals) > 1 else 1.0

                output_data = []

                if config['json_type'] == 'vector':
                    data_vars = list(subset.data_vars)
                    if len(data_vars) < 2: continue
                    var1, var2 = subset[data_vars[0]], subset[data_vars[1]]
                    flat1 = [round(float(x), 1) for x in np.where(np.isnan(var1.values), 0, var1.values).flatten()]
                    flat2 = [round(float(x), 1) for x in np.where(np.isnan(var2.values), 0, var2.values).flatten()]
                    output_data.append(make_record(lon_vals, lat_vals, dx, dy, 2, flat1, ref_time_iso))
                    output_data.append(make_record(lon_vals, lat_vals, dx, dy, 3, flat2, ref_time_iso))
                else:
                    data_vars = list(subset.data_vars)
                    if not data_vars: continue
                    var1 = subset[data_vars[0]]
                    precision = 1
                    if job_type in ['snow', 'smoke', 'dust']: precision = 3
                    flat1 = [round(float(x), precision) for x in np.where(np.isnan(var1.values), 0, var1.values).flatten()]
                    output_data.append(make_record(lon_vals, lat_vals, dx, dy, 0, flat1, ref_time_iso))

                # --- SAVE INTO SUBDIRECTORY ---
                with open(os.path.join(job_dir, filename), 'w') as f:
                    json.dump(output_data, f)
                count += 1

        ds.close()
        return True, ref_time_iso, count

    except Exception as e:
        print(f"❌ Error tiling {job_type}: {e}")
        return False, None, 0

def make_record(lon, lat, dx, dy, p_num, data, ref_time):
    return {
        "header": {
            "lo1": float(lon[0]), "la1": float(lat[0]),
            "dx": float(dx), "dy": float(dy),
            "nx": len(lon), "ny": len(lat),
            "parameterNumber": p_num,
            "refTime": ref_time
        },
        "data": data
    }

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
            success, ref_time, count = generate_tiles(grib_path, forecast_hour, job_type, conf)
            if os.path.exists(grib_path): os.remove(grib_path)
            return ref_time, count
        else:
            print(f"⚠️ NOAA Error {response.status_code}")
            return None, 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None, 0

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
        for hour in HOURS_TO_FETCH:
            ref_time, count = download_and_process(job, date, run, hour)
            if ref_time: job_ref_time = ref_time
            total_tiles += count
            time.sleep(1)
        
        if job_ref_time:
            manifest_updates[job] = {"ref_time": job_ref_time, "tiles": total_tiles}

    if manifest_updates:
        print("\n✅ Batch Complete.")
        with open(os.path.join(OUTPUT_DIR, "manifest_update.json"), 'w') as f:
            json.dump(manifest_updates, f, indent=2)
