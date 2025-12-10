import os
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
        "model": "gfs_0p25",
        "base_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "vars": {"var_UGRD": "on", "var_VGRD": "on"},
        "level": {"lev_10_m_above_ground": "on"},
        "json_type": "vector",
        "name_map": {"u10": "u", "v10": "v"}
    },
    "temp": {
        "model": "gfs_0p25",
        "base_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "vars": {"var_TMP": "on"},
        "level": {"lev_2_m_above_ground": "on"},
        "json_type": "scalar",
        "name_map": {"t2m": "data"} 
    },
    "humidity": {
        "model": "gfs_0p25",
        "base_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "vars": {"var_RH": "on"},
        "level": {"lev_2_m_above_ground": "on"},
        "json_type": "scalar",
        "name_map": {"r2": "data"}
    },
    "pressure": {
        "model": "gfs_0p25",
        "base_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "vars": {"var_PRMSL": "on"}, 
        "level": {"lev_mean_sea_level": "on"},
        "json_type": "scalar",
        "name_map": {"prmsl": "data"}
    },
    "waves": {
        "model": "gfs_0p25",
        "base_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "vars": {"var_HTSGW": "on", "var_WVDIR": "on"}, 
        "level": {"lev_surface": "on"},
        "json_type": "vector", 
        "name_map": {"swh": "u", "mwdir": "v"} 
    },
    "snow": {
        "model": "gfs_0p25",
        "base_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "vars": {"var_SNOD": "on"},
        "level": {"lev_surface": "on"},
        "json_type": "scalar",
        "name_map": {"sde": "data"} 
    },
    "cape": {
        "model": "gfs_0p25",
        "base_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "vars": {"var_CAPE": "on"},
        "level": {"lev_surface": "on"},
        "json_type": "scalar",
        "name_map": {"cape": "data"}
    },
    # Note: GEFS Aerosol URLs often change or have different server stability.
    # If this fails in Actions, we might need a fallback or check the URL.
    "smoke": {
        "model": "gefs_aero", 
        "base_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_aer_0p50.pl",
        "vars": {"var_PMTF": "on"}, 
        "level": {"lev_surface": "on"},
        "json_type": "scalar",
        "name_map": {"pmtf": "data"} 
    },
    "dust": {
        "model": "gefs_aero",
        "base_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_aer_0p50.pl",
        "vars": {"var_DUST": "on"}, 
        "level": {"lev_surface": "on"},
        "json_type": "scalar",
        "name_map": {"dust": "data"}
    }
}

OUTPUT_DIR = "public/data"
# Fetching only hour 0 to save CI time. Add [3, 6, 9...] for real forecast.
HOURS_TO_FETCH = [0] 
TILE_SIZE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_run_time():
    now = datetime.datetime.now(datetime.timezone.utc)
    possible_runs = [0, 6, 12, 18]
    # Backdate 4.5h to ensure data is present on NOMADS
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
        # engine='cfgrib' requires eccodes installed in the runner
        ds = xr.open_dataset(grib_path, engine='cfgrib')
        
        # Normalize longitude to -180..180
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
        ref_time_iso = str(ds.time.values)

        print(f"   ✂️ Tiling {job_type}...")

        count = 0
        for lat_start in range(-90, 90, TILE_SIZE):
            for lon_start in range(-180, 180, TILE_SIZE):
                
                lat_end = min(lat_start + TILE_SIZE, 90)
                lon_end = min(lon_start + TILE_SIZE, 180)

                # Slice subset
                subset = ds.sel(
                    latitude=slice(lat_end, lat_start), 
                    longitude=slice(lon_start, lon_end)
                )

                if subset.latitude.size == 0 or subset.longitude.size == 0:
                    continue

                # File naming convention: wind_0h_N40_W020.json
                lat_label = f"N{abs(lat_start)}" if lat_start >= 0 else f"S{abs(lat_start)}"
                lon_label = f"E{abs(lon_start)}" if lon_start >= 0 else f"W{abs(lon_start)}"
                filename = f"{job_type}_{forecast_hour}h_{lat_label}_{lon_label}.json"

                lat_vals = subset.latitude.values
                lon_vals = subset.longitude.values
                
                # Dynamic grid resolution calculation
                dy = (lat_vals[-1] - lat_vals[0]) / (len(lat_vals) - 1) if len(lat_vals) > 1 else 1.0
                dx = (lon_vals[-1] - lon_vals[0]) / (len(lon_vals) - 1) if len(lon_vals) > 1 else 1.0

                output_data = []

                if config['json_type'] == 'vector':
                    # Vector logic (Wind, Waves)
                    data_vars = list(subset.data_vars)
                    if len(data_vars) < 2: continue

                    var1 = subset[data_vars[0]]
                    var2 = subset[data_vars[1]]
                    
                    flat1 = np.where(np.isnan(var1.values), 0, var1.values).flatten()
                    flat2 = np.where(np.isnan(var2.values), 0, var2.values).flatten()
                    
                    # Rounding to 1 decimal place for size optimization
                    flat1 = [round(float(x), 1) for x in flat1]
                    flat2 = [round(float(x), 1) for x in flat2]

                    # 2 = U component / Height, 3 = V component / Direction
                    output_data.append(make_record(lon_vals, lat_vals, dx, dy, 2, flat1, ref_time_iso))
                    output_data.append(make_record(lon_vals, lat_vals, dx, dy, 3, flat2, ref_time_iso))

                else:
                    # Scalar logic (Temp, Rain, etc)
                    data_vars = list(subset.data_vars)
                    if not data_vars: continue
                    
                    var1 = subset[data_vars[0]]
                    flat1 = np.where(np.isnan(var1.values), 0, var1.values).flatten()
                    
                    # Precision handling
                    precision = 1
                    if job_type == 'snow': precision = 3
                    if job_type in ['smoke', 'dust']: precision = 4
                    
                    flat1 = [round(float(x), precision) for x in flat1]
                    
                    # 0 = Scalar value
                    output_data.append(make_record(lon_vals, lat_vals, dx, dy, 0, flat1, ref_time_iso))

                with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
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
    if job_type not in NOAA_CONFIG:
        print(f"⚠️ Unknown job type: {job_type}")
        return None, 0
        
    conf = NOAA_CONFIG[job_type]
    f_str = f"{forecast_hour:03d}"
    
    # Handle model URL differences
    if conf['model'] == 'gefs_aero':
        # GEFS Aerosol: gec00.t06z.pgrb2a.0p50.f000
        # NOTE: GEFS Aerosol 0.50 is the standard distribution
        server_file = f"gec00.t{run_hour}z.pgrb2a.0p50.f{f_str}"
        dir_path = f"/gefs.{date}/{run_hour}/chem/pgrb2a.0p50"
    else:
        # GFS 0.25 is the standard distribution
        server_file = f"gfs.t{run_hour}z.pgrb2.0p25.f{f_str}"
        dir_path = f"/gfs.{date}/{run_hour}/atmos"

    grib_filename = f"temp_{job_type}_{forecast_hour}.grib2"
    grib_path = os.path.join(OUTPUT_DIR, grib_filename)

    params = {
        'file': server_file,
        'dir': dir_path,
        # Fetch GLOBAL data (0 to 360) so we can tile it locally
        'leftlon': '0', 'rightlon': '360',
        'toplat': '90', 'bottomlat': '-90',
        **conf['vars'], 
        **conf['level']
    }

    try:
        print(f"⬇️ {job_type.upper()}: Downloading f{f_str}...")
        # 180s timeout is generous for CI runners
        response = requests.get(conf['base_url'], params=params, stream=True, timeout=180)
        
        if response.status_code == 200:
            with open(grib_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    f.write(chunk)
            
            success, ref_time, count = generate_tiles(grib_path, forecast_hour, job_type, conf)
            
            if os.path.exists(grib_path): os.remove(grib_path)
            
            if success:
                print(f"   ✅ Generated {count} tiles for {job_type}")
            return ref_time, count
        else:
            print(f"⚠️ NOAA Error {response.status_code} for {job_type} (File: {server_file})")
            
    except Exception as e:
        print(f"❌ Exception {job_type}: {e}")
    
    return None, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="Data type to fetch", default="all")
    args = parser.parse_args()

    date, run = get_latest_run_time()
    print(f"--- Starting Fetch: {date} {run}z ---")

    jobs = []
    if args.type == "all":
        jobs = list(NOAA_CONFIG.keys())
    elif args.type in NOAA_CONFIG:
        jobs = [args.type]
    else:
        print(f"Invalid type. Options: {list(NOAA_CONFIG.keys())}")
        sys.exit(1)

    manifest_updates = {}
    
    for job in jobs:
        print(f"\n--- Processing: {job} ---")
        job_ref_time = None
        total_tiles = 0
        
        for hour in HOURS_TO_FETCH:
            ref_time, count = download_and_process(job, date, run, hour)
            if ref_time: 
                job_ref_time = ref_time
            total_tiles += count
            
            # Sleep 2s between downloads to be nice to NOAA servers
            time.sleep(2)
        
        if job_ref_time:
            manifest_updates[job] = {
                "ref_time": job_ref_time,
                "tiles_generated": total_tiles
            }

    if manifest_updates:
        print("\n✅ Batch Complete.")
        # In a real app, you'd merge this with the existing manifest.json
        # For simplicity, we just dump what we did this run.
        with open(os.path.join(OUTPUT_DIR, "manifest_update.json"), 'w') as f:
            json.dump(manifest_updates, f, indent=2)
