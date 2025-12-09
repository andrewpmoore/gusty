import os
import requests
import datetime
import time
import json
import numpy as np
import xarray as xr

# --- CONFIGURATION ---
BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
OUTPUT_DIR = "public/data"
# For this demo, we only fetch 'now' (0h) to save GitHub Action time. 
# You can add [3, 6, 9...] later.
HOURS_TO_FETCH = [0, 3, 6, 9, 12, 15, 18, 21, 24, 36, 48, 72]
TILE_SIZE = 20  # Size of each tile in degrees (20x20 deg)

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

def generate_tiles(grib_path, forecast_hour):
    try:
        ds = xr.open_dataset(grib_path, engine='cfgrib')
        
        # Standardize longitude to -180 to 180 for easier map tiling
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')

        ref_time_iso = str(ds.time.values)
        
        # Iterate over the world in 20-degree chunks
        for lat_start in range(-90, 90, TILE_SIZE):
            for lon_start in range(-180, 180, TILE_SIZE):
                
                # Define bounds
                lat_end = min(lat_start + TILE_SIZE, 90)
                lon_end = min(lon_start + TILE_SIZE, 180)

                # Slice the dataset (Add small buffer for seamless stitching)
                # Note: GFS lat usually goes High->Low, so we slice Max to Min
                subset = ds.sel(
                    latitude=slice(lat_end, lat_start), 
                    longitude=slice(lon_start, lon_end)
                )

                # Skip empty tiles (e.g., edges)
                if subset.latitude.size == 0 or subset.longitude.size == 0:
                    continue

                # Filename scheme: wind_0h_N40_W020.json
                lat_label = f"N{abs(lat_start)}" if lat_start >= 0 else f"S{abs(lat_start)}"
                lon_label = f"E{abs(lon_start)}" if lon_start >= 0 else f"W{abs(lon_start)}"
                filename = f"wind_{forecast_hour}h_{lat_label}_{lon_label}.json"
                
                # --- Convert subset to JSON ---
                u_var = subset['u10']
                v_var = subset['v10']
                lat = subset.latitude.values
                lon = subset.longitude.values
                
                # Grid spacing
                dy = (lat[-1] - lat[0]) / (len(lat) - 1) if len(lat) > 1 else 1.0
                dx = (lon[-1] - lon[0]) / (len(lon) - 1) if len(lon) > 1 else 1.0

                output_data = []
                for name, var, param_num in [("UGRD", u_var, 2), ("VGRD", v_var, 3)]:
                    flat_data = np.where(np.isnan(var.values), 0, var.values).flatten()
                    # Round to 1 decimal for compression
                    flat_data = [round(float(x), 1) for x in flat_data]

                    record = {
                        "header": {
                            "lo1": float(lon[0]),
                            "la1": float(lat[0]),
                            "dx": float(dx),
                            "dy": float(dy),
                            "nx": len(lon),
                            "ny": len(lat),
                            "parameterNumber": param_num,
                        },
                        "data": flat_data
                    }
                    output_data.append(record)

                # Save Tile
                with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
                    json.dump(output_data, f)

        ds.close()
        return True, ref_time_iso

    except Exception as e:
        print(f"❌ Error tiling: {e}")
        return False, None

def download_and_process(date, run_hour, forecast_hour):
    f_str = f"{forecast_hour:03d}"
    grib_filename = f"temp_{forecast_hour}.grib2"
    grib_path = os.path.join(OUTPUT_DIR, grib_filename)

    # Download FULL 0.25 RESOLUTION
    params = {
        'file': f"gfs.t{run_hour}z.pgrb2.0p25.f{f_str}",
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
        print(f"⬇️ Downloading Full GFS 0.25 (This is large)...")
        response = requests.get(BASE_URL, params=params, stream=True, timeout=300)
        if response.status_code == 200:
            with open(grib_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    f.write(chunk)
            
            print("✂️ Generating tiles...")
            success, ref_time = generate_tiles(grib_path, forecast_hour)
            
            if os.path.exists(grib_path):
                os.remove(grib_path)
            
            return ref_time
        else:
            print(f"⚠️ NOAA Error {response.status_code}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
    
    return None

def main():
    date, run = get_latest_run_time()
    print(f"--- Processing Tiled GFS Run: {date} {run}z ---")
    
    model_ref_time = None
    for hour in HOURS_TO_FETCH:
        ref_time = download_and_process(date, run, hour)
        if ref_time:
            model_ref_time = ref_time

    if model_ref_time:
        manifest = {
            "model_run_iso": model_ref_time,
            "generated_at": datetime.datetime.now().isoformat(),
            "tiles_available": True,
            "tile_size": TILE_SIZE
        }
        with open(os.path.join(OUTPUT_DIR, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)
        print("✅ Tiles Generated.")

if __name__ == "__main__":
    main()
