import os
import requests
import datetime
import time
import json
import numpy as np
import xarray as xr

# --- CONFIGURATION ---
# UPDATED: Changed from 1p00 to 0p50 (0.5 degree resolution)
BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl"
OUTPUT_DIR = "public/data"
HOURS_TO_FETCH = range(0, 25, 3) 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_run_time():
    """Calculates the latest likely available GFS run."""
    now = datetime.datetime.now(datetime.timezone.utc)
    possible_runs = [0, 6, 12, 18]
    check_time = now - datetime.timedelta(hours=4) 
    
    date_str = check_time.strftime("%Y%m%d")
    hour = check_time.hour
    run_hour = max([h for h in possible_runs if h <= hour], default=18)
    
    if run_hour == 18 and hour < 4:
        prev_day = check_time - datetime.timedelta(days=1)
        date_str = prev_day.strftime("%Y%m%d")

    return date_str, f"{run_hour:02d}"

def convert_to_json(grib_path, json_path):
    try:
        ds = xr.open_dataset(grib_path, engine='cfgrib')
        u_var = ds['u10']
        v_var = ds['v10']

        lat = ds.latitude.values
        lon = ds.longitude.values
        # Dynamically calculate grid spacing so the app adapts automatically
        dy = (lat[-1] - lat[0]) / (len(lat) - 1)
        dx = (lon[-1] - lon[0]) / (len(lon) - 1)
        
        ref_time_iso = str(ds.time.values)
        output_data = []

        for name, var, param_num in [("UGRD", u_var, 2), ("VGRD", v_var, 3)]:
            flat_data = np.where(np.isnan(var.values), 0, var.values).flatten()
            flat_data = [round(float(x), 1) for x in flat_data]

            record = {
                "header": {
                    "parameterNumber": param_num,
                    "lo1": float(lon[0]),
                    "la1": float(lat[0]),
                    "dx": float(dx),
                    "dy": float(dy),
                    "nx": len(lon),
                    "ny": len(lat),
                    "refTime": ref_time_iso,
                    "forecastTime": 0 
                },
                "data": flat_data
            }
            output_data.append(record)

        with open(json_path, 'w') as f:
            json.dump(output_data, f)
        
        ds.close()
        return True, ref_time_iso

    except Exception as e:
        print(f"❌ Error converting {grib_path}: {e}")
        return False, None

def download_and_process(date, run_hour, forecast_hour):
    f_str = f"{forecast_hour:03d}"
    grib_filename = f"temp_{forecast_hour}.grib2"
    json_filename = f"wind_{forecast_hour}h.json"
    grib_path = os.path.join(OUTPUT_DIR, grib_filename)
    json_path = os.path.join(OUTPUT_DIR, json_filename)

    params = {
        # UPDATED: Changed file parameter to 0p50
        'file': f"gfs.t{run_hour}z.pgrb2.0p50.f{f_str}",
        'lev_10_m_above_ground': 'on',
        'var_UGRD': 'on',
        'var_VGRD': 'on',
        'leftlon': '0',
        'rightlon': '360',
        'toplat': '90',
        'bottomlat': '-90',
        'dir': f"/gfs.{date}/{run_hour}/atmos"
    }

    try:
        print(f"⬇️ Downloading f{f_str} (0.50 deg)...")
        response = requests.get(BASE_URL, params=params, stream=True, timeout=120) # Increased timeout for larger files
        if response.status_code == 200:
            with open(grib_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            success, ref_time = convert_to_json(grib_path, json_path)
            if os.path.exists(grib_path):
                os.remove(grib_path)
            
            if success:
                return {
                    "hour": forecast_hour,
                    "file": json_filename,
                    "ref_time": ref_time
                }
        else:
            print(f"⚠️ NOAA Error {response.status_code} for f{f_str}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
    
    return None

def main():
    date, run = get_latest_run_time()
    print(f"--- Processing GFS Run: {date} {run}z (0.50 Resolution) ---")
    
    generated_files = []
    model_ref_time = None

    for hour in HOURS_TO_FETCH:
        result = download_and_process(date, run, hour)
        if result:
            generated_files.append(result)
            model_ref_time = result['ref_time']
        # Added small sleep to prevent NOAA throttling
        time.sleep(1)

    if generated_files:
        manifest = {
            "model_run_iso": model_ref_time,
            "generated_at": datetime.datetime.now().isoformat(),
            "files": generated_files
        }
        
        manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print("✅ Manifest generated.")
    else:
        print("❌ No files generated.")
        exit(1)

if __name__ == "__main__":
    main()
