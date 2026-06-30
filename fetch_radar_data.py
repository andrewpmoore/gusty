import argparse
import bz2
import datetime as dt
import gzip
import io
import json
import math
import os
import re
import shutil
import struct
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlencode

import numpy as np
import requests
from PIL import Image, ImageSequence


OUTPUT_DIR = "public/data"
RADAR_DIR = os.path.join(OUTPUT_DIR, "radar")
RADAR_VECTOR_DIR = os.path.join(OUTPUT_DIR, "radar_vectors")
LAYER_DIR = os.path.join(OUTPUT_DIR, "layers")

RADAR_RAIN_RATE_ID = "radar_rain_rate"
RADAR_SNOW_RATE_ID = "radar_snow_rate"
RADAR_MIXED_RATE_ID = "radar_mixed_rate"
RADAR_PRECIP_RATE_ID = "radar_precip_rate"
RADAR_SCALAR_LAYER_IDS = [
    RADAR_RAIN_RATE_ID,
    RADAR_SNOW_RATE_ID,
    RADAR_MIXED_RATE_ID,
    RADAR_PRECIP_RATE_ID,
]

FORECAST_OFFSETS_MINUTES = [0, 15, 30, 45, 60, 75, 90, 105, 120]
RADAR_SOURCE_REFERENCE = "official and public meteorological radar provider references"
RADAR_VALUE_PARAMETER = "rain_rate"
RADAR_VALUE_UNITS = "mm/h"
RADAR_PROVIDER_REGISTRY_FILE = os.getenv("RADAR_PROVIDER_REGISTRY_FILE", "radar_provider_registry.json")
RADAR_PROVIDER_REGISTRY_URL = os.getenv("RADAR_PROVIDER_REGISTRY_URL", "").strip()


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


RADAR_ALLOW_UNVERIFIED_PROVIDERS = env_bool("RADAR_ALLOW_UNVERIFIED_PROVIDERS", False)
TIMEOUT = int(os.getenv("RADAR_HTTP_TIMEOUT", "90"))
USER_AGENT = os.getenv(
    "GUSTY_USER_AGENT",
    "GustyWeather/1.0 (https://gustyweather.com; weather-radar-ingestion)",
)

RADAR_TARGET_DEGREES = float(os.getenv("RADAR_TARGET_DEGREES", "0.025") or "0.025")
RADAR_MIN_VALUE = float(os.getenv("RADAR_MIN_VALUE", "0.08") or "0.08")
RADAR_NOISE_FLOOR_MM_H = float(os.getenv("RADAR_NOISE_FLOOR_MM_H", "0.08") or "0.08")
RADAR_EDGE_THRESHOLD_MM_H = float(os.getenv("RADAR_EDGE_THRESHOLD_MM_H", "0.04") or "0.04")
RADAR_SPECKLE_MAX_VALUE_MM_H = float(os.getenv("RADAR_SPECKLE_MAX_VALUE_MM_H", "0.8") or "0.8")
RADAR_SPECKLE_MIN_NEIGHBORS = int(os.getenv("RADAR_SPECKLE_MIN_NEIGHBORS", "2") or "2")
RADAR_SMOOTHING_PASSES = int(os.getenv("RADAR_SMOOTHING_PASSES", "2") or "2")
RADAR_DILATION_PASSES = int(os.getenv("RADAR_DILATION_PASSES", "1") or "1")
PERSISTENCE_END_MINUTES = int(os.getenv("RADAR_PERSISTENCE_END_MINUTES", "135") or "135")
MODEL_BLEND_START_MINUTES = int(os.getenv("RADAR_MODEL_BLEND_START_MINUTES", "45") or "45")
ENABLE_GLOBAL_MODEL_FALLBACK = env_bool("RADAR_ENABLE_GLOBAL_MODEL_FALLBACK", True)
NOAA_GFS_FORECAST_HOURS = [
    int(value.strip())
    for value in os.getenv("NOAA_GFS_FORECAST_HOURS", "0,1,2").split(",")
    if value.strip()
]

PRECIP_TYPE_SNOW_WET_BULB_C = float(os.getenv("RADAR_SNOW_WET_BULB_C", "0.0") or "0.0")
PRECIP_TYPE_RAIN_WET_BULB_C = float(os.getenv("RADAR_RAIN_WET_BULB_C", "1.5") or "1.5")

TILE_SIZE = 20
PACK_LAT_SIZE = 60
PACK_LON_SIZE = 80
OVERVIEW_KEY_SUFFIX = "_60"
OVERVIEW_SAMPLE_STEP = 4

MAGIC = b"GSTY"
VERSION = 1
ENCODING_INT16_QUANTIZED = 1
INT16_MIN_VALUE = -32767
INT16_MAX_VALUE = 32767
PACK_MAGIC = b"GPAK"
PACK_VERSION = 1
PARAMETER_RAIN_RATE = 0

TILE_FORMAT = {
    "name": "gusty-grid",
    "version": 1,
    "extension": "gtile",
    "encoding": "int16-quantized",
    "endianness": "little",
    "missing_value": -32768,
    "decode_formula": "value = offset + raw * scale",
}
PACK_FORMAT = {
    "name": "gusty-pack",
    "version": 1,
    "extension": "gpack",
    "lat_size": PACK_LAT_SIZE,
    "lon_size": PACK_LON_SIZE,
    "contains": TILE_FORMAT["extension"],
    "overview_key_suffix": OVERVIEW_KEY_SUFFIX,
    "overview_sample_step": OVERVIEW_SAMPLE_STEP,
    "offset_base": "start_of_data_section",
}

RAINBOW_RAIN_COLOR_STOPS = [
    {"value": 0.08, "color": "#7db7ff", "alpha": 0.0, "label": "transparent_edge"},
    {"value": 0.2, "color": "#5fa8dc", "alpha": 0.35, "label": "trace"},
    {"value": 1.0, "color": "#3b82c4", "alpha": 0.58, "label": "light"},
    {"value": 4.0, "color": "#6d6ac8", "alpha": 0.68, "label": "moderate"},
    {"value": 10.0, "color": "#e94d83", "alpha": 0.75, "label": "heavy"},
    {"value": 24.0, "color": "#ff5f5f", "alpha": 0.82, "label": "very_heavy"},
    {"value": 48.0, "color": "#ff9e43", "alpha": 0.88, "label": "extreme"},
    {"value": 80.0, "color": "#ffff33", "alpha": 0.95, "label": "violent"},
]
RAINBOW_SNOW_COLOR_STOPS = [
    {"value": 0.08, "color": "#f0fbff", "alpha": 0.0, "label": "transparent_edge"},
    {"value": 0.2, "color": "#dff6ff", "alpha": 0.35, "label": "trace"},
    {"value": 1.0, "color": "#a9e8ff", "alpha": 0.55, "label": "light"},
    {"value": 4.0, "color": "#72d2ff", "alpha": 0.68, "label": "moderate"},
    {"value": 10.0, "color": "#2ba7e0", "alpha": 0.76, "label": "heavy"},
    {"value": 24.0, "color": "#0f77bf", "alpha": 0.85, "label": "extreme"},
]
RAINBOW_MIXED_COLOR_STOPS = [
    {"value": 0.08, "color": "#fff1ff", "alpha": 0.0, "label": "transparent_edge"},
    {"value": 0.2, "color": "#f4d9ff", "alpha": 0.35, "label": "trace"},
    {"value": 1.0, "color": "#d8a6ff", "alpha": 0.56, "label": "light"},
    {"value": 4.0, "color": "#b667e7", "alpha": 0.7, "label": "moderate"},
    {"value": 10.0, "color": "#8d31c7", "alpha": 0.78, "label": "heavy"},
    {"value": 24.0, "color": "#5a148a", "alpha": 0.86, "label": "extreme"},
]

REGION_BOUNDS = {
    "North America": (-170.0, 5.0, -30.0, 85.0),
    "Canada": (-141.0, 41.5, -52.5, 83.5),
    "United States": (-126.0, 24.0, -66.0, 50.0),
    "Germany": (4.5, 46.8, 15.5, 55.5),
    "Finland": (18.0, 58.0, 33.5, 71.5),
    "Japan": (122.8, 24.0, 146.1, 46.1),
    "El Salvador": (-90.833, 12.112, -87.044, 15.244),
    "Taiwan": (115.0, 18.0, 126.5125, 29.0125),
    "Malaysia Peninsular": (96.92, -1.33, 106.28, 8.97),
    "Malaysia East": (107.08, -1.48, 121.19, 9.18),
    "Europe OPERA": (-25.0, 34.0, 45.0, 72.0),
    "Italy DPC": (4.5, 35.0, 19.1, 47.6),
}

ECCC_GEOMET_URL = "https://geo.weather.gc.ca/geomet"
ECCC_RADAR_DOCS_URL = "https://eccc-msc.github.io/open-data/msc-data/obs_radar/readme_radar_geomet_en/"
NOAA_MRMS_BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
NOAA_MRMS_DOCS_URL = "https://registry.opendata.aws/noaa-mrms/"
NOAA_MRMS_PRODUCT_PREFIX = "CONUS/PrecipRate_00.00"
NOAA_GFS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
DWD_RADAR_RV_URL = "https://opendata.dwd.de/weather/radar/composite/rv/"
DWD_RADAR_RS_URL = "https://opendata.dwd.de/weather/radar/composite/rs/"
DWD_SOURCE_URL = "https://opendata.dwd.de/weather/radar/composite/"
FMI_WMS_URL = "https://openwms.fmi.fi/geoserver/wms"
FMI_OPEN_DATA_URL = "https://en.ilmatieteenlaitos.fi/open-data"
FMI_LICENSE_URL = "https://en.ilmatieteenlaitos.fi/open-data-licence"
FMI_RAIN_RATE_LAYER = "Radar:radar_finland_cappi_rate"
JMA_NOWCAST_TIMES_URL = "https://www.jma.go.jp/bosai/jmatile/data/nowc/targetTimes_N1.json"
JMA_NOWCAST_TILE_TEMPLATE = (
    "https://www.jma.go.jp/bosai/jmatile/data/nowc/{base_time}/none/"
    "{valid_time}/surf/hrpns/{z}/{x}/{y}.png"
)
MARN_BUCKET_URL = "https://storage.googleapis.com"
MARN_BUCKET_API_URL = "https://storage.googleapis.com/storage/v1/b/radar-images-sv/o"
MARN_PRODUCT_PREFIX = "esar82/Images/"
CWA_BASE_URL = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com"
CWA_ARCHIVE_PREFIX = "/history/Observation"
MMD_RADAR_GIF_URL = "https://api.met.gov.my/static/images/radar-latest.gif"
OPERA_BASE_URL = "https://s3.waw3-1.cloudferro.com/openradar-24h"
DPC_API_BASE_URL = "https://radar-api.protezionecivile.it"
NOAA_MRMS_NCEP_BASE_URL = "https://mrms.ncep.noaa.gov/2D"
IEM_BASE_URL = "https://mesonet.agron.iastate.edu"

ECCC_RAIN_RAMP = [
    (0.1, "#8cc7fe"),
    (1.0, "#40aefe"),
    (2.0, "#00e092"),
    (4.0, "#00d615"),
    (8.0, "#009d00"),
    (12.0, "#006600"),
    (16.0, "#fef800"),
    (24.0, "#fec100"),
    (32.0, "#fe8700"),
    (50.0, "#fe3700"),
    (64.0, "#fe0159"),
    (100.0, "#ba23ba"),
    (125.0, "#6e08a1"),
    (200.0, "#33004d"),
]
ECCC_SNOW_RAMP = [
    (0.1, "#eff8ff"),
    (0.5, "#c7e9ff"),
    (1.0, "#8fd4ff"),
    (2.0, "#4eb8ed"),
    (4.0, "#1584d1"),
    (8.0, "#075aaa"),
    (16.0, "#063970"),
]
FMI_RAIN_RAMP = [
    (0.0, "#e5d3f7"),
    (0.1, "#cbbef4"),
    (0.5, "#aeaef2"),
    (1.0, "#93a7f0"),
    (2.5, "#7db0ef"),
    (5.0, "#64c4ee"),
    (7.5, "#58dee2"),
    (10.0, "#95ed8d"),
    (15.0, "#e0df2b"),
    (20.0, "#e1be12"),
    (25.0, "#cf930b"),
    (30.0, "#ba6d08"),
    (40.0, "#a64b05"),
    (65.0, "#8f2f03"),
    (100.0, "#781501"),
]
JMA_HRPNS_COLOR_VALUES = {
    "#f2f2ff": 0.5,
    "#a0d2ff": 3.0,
    "#218cff": 7.5,
    "#0041ff": 15.0,
    "#faf500": 25.0,
    "#ff9900": 40.0,
    "#ff2800": 65.0,
    "#b40068": 90.0,
}
MMD_DBZ_PALETTE = (
    (210, 10, 210, 65.0),
    (255, 55, 255, 64.6),
    (255, 115, 255, 62.7),
    (180, 0, 0, 59.8),
    (220, 0, 0, 55.0),
    (255, 38, 0, 53.5),
    (247, 119, 0, 50.2),
    (247, 165, 0, 43.8),
    (255, 209, 0, 39.0),
    (255, 255, 0, 37.5),
    (0, 240, 0, 34.2),
    (0, 200, 0, 27.8),
    (0, 172, 0, 23.0),
    (0, 135, 0, 21.5),
    (52, 206, 236, 18.2),
    (5, 155, 255, 11.8),
    (0, 113, 226, 7.0),
    (255, 255, 255, 2.2),
)
MMD_MAX_RGB_DIST2 = int(os.getenv("MMD_MAX_RGB_DIST2", "64") or "64")
MMD_SUBRECTS = {
    "MYPENINSULAR": (11, 562, 0, 424),
    "MYEAST": (0, 570, 460, 1100),
}
MRMS_REFLECTIVITY_PRODUCTS = {
    "USCOMP": ("MergedReflectivityQCComposite", (-129.995, 20.005, -60.005, 54.995)),
    "AKCOMP": ("ALASKA/MergedReflectivityQCComposite", (-175.995, 50.005, -126.005, 71.995)),
    "HICOMP": ("HAWAII/MergedReflectivityQCComposite", (-163.998, 15.002, -151.002, 25.997)),
    "PRCOMP": ("CARIB/MergedReflectivityQCComposite", (-89.995, 10.005, -60.005, 24.995)),
    "GUCOMP": ("GUAM/MergedReflectivityQCComposite", (140.002, 9.002, 149.998, 17.997)),
}
IEM_REGIONS = {
    "USCOMP": ("USCOMP", (-126.0, 23.0, -65.0, 50.0)),
    "AKCOMP": ("AKCOMP", (-170.5, 53.2, -130.5, 68.7)),
    "HICOMP": ("HICOMP", (-162.4, 15.4, -152.4, 24.4)),
    "PRCOMP": ("PRCOMP", (-71.1, 13.1, -61.1, 23.1)),
    "GUCOMP": ("GUCOMP", (140.5, 9.2, 149.0, 17.7)),
}


@dataclass
class PrecipFrame:
    provider_id: str
    name: str
    valid_time: Optional[str]
    lead_minutes: int
    bounds: Tuple[float, float, float, float]
    latitudes: np.ndarray
    longitudes: np.ndarray
    rain_mm_h: np.ndarray
    snow_mm_h: Optional[np.ndarray]
    mixed_mm_h: Optional[np.ndarray]
    attribution: str
    source_url: str
    source_kind: str
    machine_readable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResult:
    id: str
    name: str
    status: str
    source_url: str
    attribution: str
    frames: List[PrecipFrame] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkGrid:
    rain: np.ndarray
    snow: np.ndarray
    mixed: np.ndarray
    model_active: bool
    source_provider_ids: List[str]
    source_attributions: List[str]
    source_urls: List[str]
    ref_times: List[str]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def http_session() -> requests.Session:
    client = requests.Session()
    client.headers.update({"User-Agent": USER_AGENT})
    return client


def ensure_clean_output() -> None:
    for layer_id in RADAR_SCALAR_LAYER_IDS:
        shutil.rmtree(os.path.join(OUTPUT_DIR, layer_id), ignore_errors=True)
        os.makedirs(os.path.join(OUTPUT_DIR, layer_id), exist_ok=True)
    shutil.rmtree(RADAR_DIR, ignore_errors=True)
    os.makedirs(RADAR_DIR, exist_ok=True)
    shutil.rmtree(RADAR_VECTOR_DIR, ignore_errors=True)
    os.makedirs(RADAR_VECTOR_DIR, exist_ok=True)
    os.makedirs(LAYER_DIR, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=True)


def get_bytes(client: requests.Session, url: str) -> Tuple[bytes, Optional[str]]:
    response = client.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.content, parse_last_modified(response.headers)


def get_text(client: requests.Session, url: str) -> Tuple[str, Optional[str]]:
    response = client.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text, parse_last_modified(response.headers)


def parse_last_modified(headers: Dict[str, str]) -> Optional[str]:
    value = headers.get("Last-Modified") or headers.get("Date")
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def parse_iso_or_none(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError:
        return None


def coordinate_label(lat_start: int, lon_start: int) -> Tuple[str, str]:
    lat_label = f"N{abs(lat_start)}" if lat_start >= 0 else f"S{abs(lat_start)}"
    lon_label = f"E{abs(lon_start)}" if lon_start >= 0 else f"W{abs(lon_start)}"
    return lat_label, lon_label


def tile_key(lat_start: int, lon_start: int) -> str:
    lat_label, lon_label = coordinate_label(lat_start, lon_start)
    return f"{lat_label}_{lon_label}"


def pack_origin(lat_start: int, lon_start: int) -> Tuple[int, int]:
    pack_lat_start = math.floor(lat_start / PACK_LAT_SIZE) * PACK_LAT_SIZE
    pack_lat_start = max(-90, min(60, pack_lat_start))
    pack_lon_start = math.floor((lon_start + 120) / PACK_LON_SIZE) * PACK_LON_SIZE - 120
    pack_lon_start = max(-180, min(120, pack_lon_start))
    return pack_lat_start, pack_lon_start


def pack_filename_minutes(product_id: str, minutes: int, lat_start: int, lon_start: int) -> str:
    lat_label, lon_label = coordinate_label(lat_start, lon_start)
    return f"{product_id}_{minutes}m_{lat_label}_{lon_label}.{PACK_FORMAT['extension']}"


def overview_tile_key(lat_start: int, lon_start: int) -> str:
    return f"{tile_key(lat_start, lon_start)}{OVERVIEW_KEY_SUFFIX}"


def quantize_int16(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float32)
    min_value = float(np.nanmin(values)) if values.size else 0.0
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(min_value):
        min_value = 0.0
    if not np.isfinite(max_value):
        max_value = 0.0
    if min_value == max_value:
        return np.zeros(values.shape, dtype="<i2"), 1.0, min_value
    scale = (max_value - min_value) / (INT16_MAX_VALUE - INT16_MIN_VALUE)
    offset = (max_value + min_value) / 2.0
    quantized = np.rint((values - offset) / scale)
    quantized = np.clip(quantized, INT16_MIN_VALUE, INT16_MAX_VALUE).astype("<i2")
    return quantized, float(scale), float(offset)


def build_binary_tile_bytes(
    lon_vals: np.ndarray,
    lat_vals: np.ndarray,
    dx: float,
    dy: float,
    values: np.ndarray,
) -> bytes:
    flattened = np.round(np.where(np.isfinite(values), values, 0.0).astype(np.float32), 3).flatten()
    expected_value_count = len(lon_vals) * len(lat_vals)
    if flattened.size != expected_value_count:
        raise ValueError(f"Tile has {flattened.size} values, expected {expected_value_count}")

    header = bytearray()
    header.extend(MAGIC)
    header.extend(
        struct.pack(
            "<BBBBHHffff",
            VERSION,
            ENCODING_INT16_QUANTIZED,
            1,
            0,
            len(lon_vals),
            len(lat_vals),
            float(lon_vals[0]),
            float(lat_vals[0]),
            float(dx),
            float(dy),
        )
    )
    quantized, scale, offset = quantize_int16(flattened)
    header.extend(struct.pack("<hff", PARAMETER_RAIN_RATE, scale, offset))
    return bytes(header) + quantized.tobytes()


def write_tile_pack(pack_path: str, entries: List[Tuple[str, bytes]]) -> None:
    header = bytearray()
    data = bytearray()
    table = []
    for key, tile_bytes in entries:
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

    os.makedirs(os.path.dirname(pack_path), exist_ok=True)
    with open(pack_path, "wb") as handle:
        handle.write(header)
        handle.write(data)


def normalize_hex_color(value: str) -> Tuple[int, int, int]:
    cleaned = value.strip().lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color: {value}")
    return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)


def rgba_to_color_ramp_values(
    image: Image.Image,
    ramp: Sequence[Tuple[float, str]],
    max_color_distance: Optional[int] = None,
    alpha_threshold: int = 1,
) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    output = np.zeros(rgba.shape[:2], dtype=np.float32)
    mask = rgba[:, :, 3] >= alpha_threshold
    if not np.any(mask):
        return output
    colors = np.array([normalize_hex_color(color) for _, color in ramp], dtype=np.int32)
    values = np.array([value for value, _ in ramp], dtype=np.float32)
    pixels = rgba[:, :, :3][mask].astype(np.int32)
    distances = np.sum((pixels[:, None, :] - colors[None, :, :]) ** 2, axis=2)
    closest = np.argmin(distances, axis=1)
    if max_color_distance is not None:
        accepted = distances[np.arange(len(closest)), closest] <= max_color_distance
        mapped = np.zeros(len(closest), dtype=np.float32)
        mapped[accepted] = values[closest[accepted]]
    else:
        mapped = values[closest]
    output[mask] = mapped
    return np.where(output >= RADAR_MIN_VALUE, output, 0.0).astype(np.float32)


def rgba_to_configured_values(image: Image.Image, color_values: Dict[str, Any]) -> np.ndarray:
    ramp = [(float(value), color) for color, value in color_values.items()]
    return rgba_to_color_ramp_values(image, ramp, max_color_distance=int(os.getenv("RADAR_CONFIG_COLOR_DISTANCE", "900")))


def dbz_to_mm_h(dbz: np.ndarray) -> np.ndarray:
    values = np.asarray(dbz, dtype=np.float32)
    z = np.power(10.0, values / 10.0)
    rain = np.power(z / 200.0, 1.0 / 1.6)
    return np.where(np.isfinite(rain) & (values > -32.0), np.clip(rain, 0.0, 250.0), 0.0).astype(np.float32)


def decode_marn_png(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    alpha = rgba[..., 3]
    hue = np.full(rgba.shape[:2], np.nan, dtype=np.float32)
    arc1 = (g == 255) & (r == 0)
    hue[arc1] = 120.0 + b[arc1].astype(np.float32) * (60.0 / 255.0)
    arc2 = (b == 255) & (r == 0) & (g != 255)
    hue[arc2] = 240.0 - g[arc2].astype(np.float32) * (60.0 / 255.0)
    arc3 = (b == 255) & (g == 0) & (r != 0)
    hue[arc3] = 240.0 + r[arc3].astype(np.float32) * (60.0 / 255.0)
    dbz = 10.0 + (hue - 120.0) * (60.0 / 180.0)
    return dbz_to_mm_h(np.where((alpha == 0) | np.isnan(hue), -33.0, dbz))


def parse_cwa_xml(image_bytes: bytes, width: int = 921, height: int = 881) -> np.ndarray:
    root = ET.fromstring(image_bytes)
    content = None
    for element in root.iter():
        if element.tag.endswith("content"):
            content = element.text
            break
    if not content:
        raise ValueError("CWA XML missing content element")
    flat = np.fromstring(content, sep=",", dtype=np.float32)
    expected = width * height
    if flat.size != expected:
        raise ValueError(f"CWA grid size mismatch: {flat.size}, expected {expected}")
    dbz = flat.reshape(height, width)[::-1]
    dbz = np.where(dbz <= -99.0, -33.0, dbz)
    return dbz_to_mm_h(dbz)


def decode_mmd_rgb(rgb: np.ndarray) -> np.ndarray:
    palette_rgb = np.array([(r, g, b) for r, g, b, _ in MMD_DBZ_PALETTE], dtype=np.int32)
    palette_dbz = np.array([dbz for *_, dbz in MMD_DBZ_PALETTE], dtype=np.float32)
    flat = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    distances = np.sum((flat[:, None, :] - palette_rgb[None, :, :]) ** 2, axis=2)
    closest = np.argmin(distances, axis=1)
    closest_distance = distances[np.arange(flat.shape[0]), closest]
    dbz = np.full(flat.shape[0], -33.0, dtype=np.float32)
    valid = closest_distance <= MMD_MAX_RGB_DIST2
    dbz[valid] = palette_dbz[closest[valid]]
    return dbz_to_mm_h(dbz.reshape(rgb.shape[:2]))


def decode_iem_n0q_png(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.mode == "P":
            raw = np.asarray(image, dtype=np.uint8)
        else:
            raw = np.asarray(image.convert("L"), dtype=np.uint8)
    dbz = raw.astype(np.float32) / 2.0 - 32.0
    return dbz_to_mm_h(np.where(raw == 0, -33.0, dbz))


def fill_thin_gaps(values: np.ndarray) -> np.ndarray:
    mask = values > 0
    padded = np.pad(mask, 1, mode="constant")
    neighbor_count = sum(
        padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    )
    gap_pixels = (~mask) & (neighbor_count >= 6)
    if not np.any(gap_pixels):
        return values
    padded_values = np.pad(values, 1, mode="constant")
    neighbor_max = np.maximum.reduce(
        [
            padded_values[1 + dy : 1 + dy + values.shape[0], 1 + dx : 1 + dx + values.shape[1]]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dy or dx
        ]
    )
    output = values.copy()
    output[gap_pixels] = neighbor_max[gap_pixels]
    return output.astype(np.float32)


def tile_axis(start: float, end: float, step: float, descending: bool = False) -> np.ndarray:
    if descending:
        values = np.arange(end, start - step * 0.5, -step, dtype=np.float64)
        return values[(values >= start) & (values <= end)]
    values = np.arange(start, end + step * 0.5, step, dtype=np.float64)
    return values[(values >= start) & (values <= end)]


def grid_bounds_from_axes(latitudes: np.ndarray, longitudes: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(np.nanmin(longitudes)),
        float(np.nanmin(latitudes)),
        float(np.nanmax(longitudes)),
        float(np.nanmax(latitudes)),
    )


def intersects_bounds(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    west_a, south_a, east_a, north_a = a
    west_b, south_b, east_b, north_b = b
    return west_a <= east_b and east_a >= west_b and south_a <= north_b and north_a >= south_b


def clamp_bounds(bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    west, south, east, north = bounds
    return max(-180.0, west), max(-90.0, south), min(180.0, east), min(90.0, north)


def bounds_tile_origins(bounds: Tuple[float, float, float, float]) -> Iterable[Tuple[int, int]]:
    west, south, east, north = clamp_bounds(bounds)
    lat_start_min = math.floor(south / TILE_SIZE) * TILE_SIZE
    lat_start_max = math.floor(north / TILE_SIZE) * TILE_SIZE
    lon_start_min = math.floor(west / TILE_SIZE) * TILE_SIZE
    lon_start_max = math.floor(east / TILE_SIZE) * TILE_SIZE
    for lat_start in range(lat_start_min, lat_start_max + TILE_SIZE, TILE_SIZE):
        if lat_start < -90 or lat_start >= 90:
            continue
        for lon_start in range(lon_start_min, lon_start_max + TILE_SIZE, TILE_SIZE):
            if lon_start < -180 or lon_start >= 180:
                continue
            yield lat_start, lon_start


def normalize_lonlat_arrays(values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    longitudes = np.where(longitudes > 180.0, longitudes - 360.0, longitudes)
    if latitudes[0] > latitudes[-1]:
        latitudes = latitudes[::-1]
        values = values[::-1, :]
    if longitudes[0] > longitudes[-1]:
        order = np.argsort(longitudes)
        longitudes = longitudes[order]
        values = values[:, order]
    return values, latitudes, longitudes


def sample_grid(
    values: np.ndarray,
    x_pixel_size: float,
    y_pixel_size: float,
    west: float,
    north: float,
    target_degrees: float = RADAR_TARGET_DEGREES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    native_resolution = max(abs(x_pixel_size), abs(y_pixel_size), 1e-6)
    sample_step = max(1, int(round(target_degrees / native_resolution)))
    sampled = values[::sample_step, ::sample_step].astype(np.float32)
    rows, cols = sampled.shape
    longitudes = west + np.arange(cols, dtype=np.float64) * x_pixel_size * sample_step
    latitudes = north + np.arange(rows, dtype=np.float64) * y_pixel_size * sample_step
    return normalize_lonlat_arrays(sampled, latitudes, longitudes)


def bilinear_interpolate_coords(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    src_values: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    src_values = np.asarray(src_values, dtype=np.float32)
    if len(src_lat) < 2 or len(src_lon) < 2:
        return np.full((len(target_lats), len(target_lons)), fill_value, dtype=np.float32)
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        src_values = src_values[::-1, :]
    if src_lon[0] > src_lon[-1]:
        order = np.argsort(src_lon)
        src_lon = src_lon[order]
        src_values = src_values[:, order]

    lat_mask = (target_lats >= src_lat[0]) & (target_lats <= src_lat[-1])
    lon_mask = (target_lons >= src_lon[0]) & (target_lons <= src_lon[-1])
    output = np.full((len(target_lats), len(target_lons)), fill_value, dtype=np.float32)
    if not np.any(lat_mask) or not np.any(lon_mask):
        return output

    target_lats_overlap = target_lats[lat_mask]
    target_lons_overlap = target_lons[lon_mask]
    src_lat_indices = np.arange(len(src_lat))
    src_lon_indices = np.arange(len(src_lon))
    lat_idx_frac = np.interp(target_lats_overlap, src_lat, src_lat_indices)
    lon_idx_frac = np.interp(target_lons_overlap, src_lon, src_lon_indices)
    y0 = np.floor(lat_idx_frac).astype(np.int32)
    y1 = np.minimum(y0 + 1, len(src_lat) - 1)
    x0 = np.floor(lon_idx_frac).astype(np.int32)
    x1 = np.minimum(x0 + 1, len(src_lon) - 1)
    wy = (lat_idx_frac - y0)[:, None]
    wx = (lon_idx_frac - x0)[None, :]
    c00 = src_values[np.ix_(y0, x0)]
    c01 = src_values[np.ix_(y0, x1)]
    c10 = src_values[np.ix_(y1, x0)]
    c11 = src_values[np.ix_(y1, x1)]
    interpolated = (
        c00 * (1.0 - wy) * (1.0 - wx)
        + c01 * (1.0 - wy) * wx
        + c10 * wy * (1.0 - wx)
        + c11 * wy * wx
    )
    output[np.ix_(lat_mask, lon_mask)] = np.where(np.isnan(interpolated), fill_value, interpolated)
    return output


def smooth_grid(values: np.ndarray, passes: int = RADAR_SMOOTHING_PASSES) -> np.ndarray:
    smoothed = np.asarray(values, dtype=np.float32).copy()
    for _ in range(max(0, passes)):
        padded = np.pad(smoothed, 1, mode="edge")
        smoothed = (
            padded[1:-1, 1:-1] * 0.25
            + (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.125
            + (padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:]) * 0.0625
        )
    return smoothed.astype(np.float32)


def soft_dilate(values: np.ndarray, passes: int = RADAR_DILATION_PASSES) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    for _ in range(max(0, passes)):
        padded = np.pad(result, 1, mode="constant")
        neighbors = np.maximum.reduce(
            [
                padded[0:-2, 0:-2],
                padded[0:-2, 1:-1],
                padded[0:-2, 2:],
                padded[1:-1, 0:-2],
                padded[1:-1, 1:-1],
                padded[1:-1, 2:],
                padded[2:, 0:-2],
                padded[2:, 1:-1],
                padded[2:, 2:],
            ]
        )
        result = np.maximum(result, neighbors * 0.45)
    return result.astype(np.float32)


def filter_speckle(values: np.ndarray) -> np.ndarray:
    cleaned = np.where(values >= RADAR_NOISE_FLOOR_MM_H, values, 0.0).astype(np.float32)
    if RADAR_SPECKLE_MIN_NEIGHBORS <= 0:
        return cleaned
    precip = cleaned > 0
    padded = np.pad(precip.astype(np.uint8), 1, mode="constant")
    neighbors = np.zeros(cleaned.shape, dtype=np.uint8)
    for row_offset in (0, 1, 2):
        for col_offset in (0, 1, 2):
            if row_offset == 1 and col_offset == 1:
                continue
            neighbors += padded[row_offset : row_offset + cleaned.shape[0], col_offset : col_offset + cleaned.shape[1]]
    speckle = (
        (cleaned > 0)
        & (cleaned <= RADAR_SPECKLE_MAX_VALUE_MM_H)
        & (neighbors < RADAR_SPECKLE_MIN_NEIGHBORS)
    )
    cleaned[speckle] = 0.0
    return cleaned


def polish_precip(values: np.ndarray) -> np.ndarray:
    cleaned = filter_speckle(values)
    cleaned = soft_dilate(cleaned)
    cleaned = smooth_grid(cleaned)
    return np.where(cleaned >= RADAR_EDGE_THRESHOLD_MM_H, cleaned, 0.0).astype(np.float32)


def resample_field(frame: PrecipFrame, field: str, lat_vals: np.ndarray, lon_vals: np.ndarray) -> np.ndarray:
    values = getattr(frame, field)
    if values is None:
        return np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float32)
    return bilinear_interpolate_coords(frame.latitudes, frame.longitudes, values, lat_vals, lon_vals)


def merge_max(target: np.ndarray, addition: np.ndarray, weight: float = 1.0) -> np.ndarray:
    if weight <= 0:
        return target
    addition = np.where(np.isfinite(addition), addition * weight, 0.0).astype(np.float32)
    return np.maximum(target, addition)


def persistence_weight(offset_minutes: int) -> float:
    if offset_minutes <= 0:
        return 1.0
    return max(0.0, 1.0 - (float(offset_minutes) / float(max(PERSISTENCE_END_MINUTES, 1))))


def model_blend_weight(offset_minutes: int) -> float:
    if offset_minutes <= MODEL_BLEND_START_MINUTES:
        return 0.0
    remaining = max(1, 120 - MODEL_BLEND_START_MINUTES)
    return min(0.65, (offset_minutes - MODEL_BLEND_START_MINUTES) / remaining)


def selected_frames_for_offset(frames: List[PrecipFrame], offset_minutes: int) -> List[Tuple[PrecipFrame, float]]:
    by_provider: Dict[str, List[PrecipFrame]] = {}
    for frame in frames:
        by_provider.setdefault(frame.provider_id, []).append(frame)

    selected: List[Tuple[PrecipFrame, float]] = []
    for provider_frames in by_provider.values():
        exact = [frame for frame in provider_frames if frame.lead_minutes == offset_minutes]
        if exact:
            selected.extend((frame, 1.0) for frame in exact)
            continue
        current = [frame for frame in provider_frames if frame.lead_minutes == 0]
        if current:
            weight = persistence_weight(offset_minutes)
            if weight > 0:
                selected.extend((frame, weight) for frame in current)
    return selected


def data_var_by_grib_name(dataset: Any, names: Iterable[str]) -> Any:
    wanted = {name.lower() for name in names}
    for data_var in dataset.data_vars.values():
        attrs = data_var.attrs
        candidates = [
            str(attrs.get("GRIB_shortName", "")),
            str(attrs.get("GRIB_name", "")),
            str(data_var.name),
        ]
        if any(candidate.lower() in wanted for candidate in candidates):
            return data_var
    raise ValueError(f"Could not find field matching {sorted(wanted)}")


def wet_bulb_temperature_c(temp_k_or_c: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    temp_c = np.where(temp_k_or_c > 150.0, temp_k_or_c - 273.15, temp_k_or_c).astype(np.float32)
    rh = np.clip(relative_humidity.astype(np.float32), 1.0, 100.0)
    wet_bulb = (
        temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return wet_bulb.astype(np.float32)


def split_precip_by_wet_bulb(precip_rate: np.ndarray, wet_bulb_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(wet_bulb_c)
    snow_mask = finite & (wet_bulb_c <= PRECIP_TYPE_SNOW_WET_BULB_C)
    mixed_mask = finite & (wet_bulb_c > PRECIP_TYPE_SNOW_WET_BULB_C) & (wet_bulb_c <= PRECIP_TYPE_RAIN_WET_BULB_C)
    rain_mask = ~finite | (wet_bulb_c > PRECIP_TYPE_RAIN_WET_BULB_C)
    rain = np.where(rain_mask, precip_rate, 0.0).astype(np.float32)
    snow = np.where(snow_mask, precip_rate, 0.0).astype(np.float32)
    mixed = np.where(mixed_mask, precip_rate, 0.0).astype(np.float32)
    return rain, snow, mixed


def split_precip_by_model_phase(
    precip_rate: np.ndarray,
    model_rain: np.ndarray,
    model_snow: np.ndarray,
    model_mixed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    precip = np.where(np.isfinite(precip_rate), precip_rate, 0.0).astype(np.float32)
    model_total = np.maximum(model_rain, 0.0) + np.maximum(model_snow, 0.0) + np.maximum(model_mixed, 0.0)
    has_phase = model_total > 0.0
    rain = np.where(has_phase, precip * np.maximum(model_rain, 0.0) / np.maximum(model_total, 1e-6), precip)
    snow = np.where(has_phase, precip * np.maximum(model_snow, 0.0) / np.maximum(model_total, 1e-6), 0.0)
    mixed = np.where(has_phase, precip * np.maximum(model_mixed, 0.0) / np.maximum(model_total, 1e-6), 0.0)
    return rain.astype(np.float32), snow.astype(np.float32), mixed.astype(np.float32)


def create_frame(
    provider_id: str,
    name: str,
    valid_time: Optional[str],
    lead_minutes: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    rain: Optional[np.ndarray],
    snow: Optional[np.ndarray],
    mixed: Optional[np.ndarray],
    attribution: str,
    source_url: str,
    source_kind: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> PrecipFrame:
    rain_values = np.zeros((len(latitudes), len(longitudes)), dtype=np.float32) if rain is None else np.asarray(rain, dtype=np.float32)
    snow_values = None if snow is None else np.asarray(snow, dtype=np.float32)
    mixed_values = None if mixed is None else np.asarray(mixed, dtype=np.float32)
    return PrecipFrame(
        provider_id=provider_id,
        name=name,
        valid_time=valid_time,
        lead_minutes=lead_minutes,
        bounds=grid_bounds_from_axes(latitudes, longitudes),
        latitudes=latitudes,
        longitudes=longitudes,
        rain_mm_h=np.where(np.isfinite(rain_values), rain_values, 0.0).astype(np.float32),
        snow_mm_h=None if snow_values is None else np.where(np.isfinite(snow_values), snow_values, 0.0).astype(np.float32),
        mixed_mm_h=None if mixed_values is None else np.where(np.isfinite(mixed_values), mixed_values, 0.0).astype(np.float32),
        attribution=attribution,
        source_url=source_url,
        source_kind=source_kind,
        metadata=metadata or {},
    )


def wms_get_map_url(
    service_url: str,
    layer: str,
    bounds: Tuple[float, float, float, float],
    *,
    style: str = "",
    width: int = 1800,
    height: int = 1000,
    time_value: Optional[str] = None,
) -> str:
    west, south, east, north = bounds
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": style,
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
        "CRS": "EPSG:4326",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "BBOX": f"{south},{west},{north},{east}",
    }
    if time_value:
        params["TIME"] = time_value
    return f"{service_url}?{urlencode(params)}"


def wms_url(
    layer: str,
    bounds: Tuple[float, float, float, float],
    *,
    style: str = "",
    width: int = 1800,
    height: int = 1000,
    time_value: Optional[str] = None,
) -> str:
    return wms_get_map_url(
        ECCC_GEOMET_URL,
        layer,
        bounds,
        style=style,
        width=width,
        height=height,
        time_value=time_value,
    )


def fetch_eccc_geomet(client: requests.Session) -> ProviderResult:
    provider_id = "eccc_geomet"
    bounds = REGION_BOUNDS["North America"]
    width = int(os.getenv("ECCC_RADAR_WIDTH", "2200") or "2200")
    height = int(os.getenv("ECCC_RADAR_HEIGHT", "1300") or "1300")
    frames: List[PrecipFrame] = []

    rain_layers = {
        0: ("RADAR_1KM_RRAI", "RADARURPPRECIPR14-LINEAR", ECCC_RAIN_RAMP),
        15: ("Radar_1km_RainPrecipRate-Extrapolation", "", ECCC_RAIN_RAMP),
        30: ("Radar_1km_RainPrecipRate-Extrapolation", "", ECCC_RAIN_RAMP),
        45: ("Radar_1km_RainPrecipRate-Extrapolation", "", ECCC_RAIN_RAMP),
        60: ("Radar_1km_RainPrecipRate-Extrapolation", "", ECCC_RAIN_RAMP),
        75: ("Radar_1km_RainPrecipRate-Extrapolation", "", ECCC_RAIN_RAMP),
        90: ("Radar_1km_RainPrecipRate-Extrapolation", "", ECCC_RAIN_RAMP),
        105: ("Radar_1km_RainPrecipRate-Extrapolation", "", ECCC_RAIN_RAMP),
        120: ("Radar_1km_RainPrecipRate-Extrapolation", "", ECCC_RAIN_RAMP),
    }
    snow_layers = {
        0: ("RADAR_1KM_RSNO", "", ECCC_SNOW_RAMP),
        15: ("Radar_1km_SnowPrecipRate-Extrapolation", "", ECCC_SNOW_RAMP),
        30: ("Radar_1km_SnowPrecipRate-Extrapolation", "", ECCC_SNOW_RAMP),
        45: ("Radar_1km_SnowPrecipRate-Extrapolation", "", ECCC_SNOW_RAMP),
        60: ("Radar_1km_SnowPrecipRate-Extrapolation", "", ECCC_SNOW_RAMP),
        75: ("Radar_1km_SnowPrecipRate-Extrapolation", "", ECCC_SNOW_RAMP),
        90: ("Radar_1km_SnowPrecipRate-Extrapolation", "", ECCC_SNOW_RAMP),
        105: ("Radar_1km_SnowPrecipRate-Extrapolation", "", ECCC_SNOW_RAMP),
        120: ("Radar_1km_SnowPrecipRate-Extrapolation", "", ECCC_SNOW_RAMP),
    }

    for offset in FORECAST_OFFSETS_MINUTES:
        rain = None
        snow = None
        ref_time = None
        source_urls = []
        try:
            layer, style, ramp = rain_layers[offset]
            url = wms_url(layer, bounds, style=style, width=width, height=height)
            image_bytes, ref_time = get_bytes(client, url)
            source_urls.append(url)
            with Image.open(io.BytesIO(image_bytes)) as image:
                rain = rgba_to_color_ramp_values(image, ramp)
        except Exception:
            rain = None
        try:
            layer, style, ramp = snow_layers[offset]
            url = wms_url(layer, bounds, style=style, width=width, height=height)
            image_bytes, snow_time = get_bytes(client, url)
            source_urls.append(url)
            if not ref_time:
                ref_time = snow_time
            with Image.open(io.BytesIO(image_bytes)) as image:
                snow = rgba_to_color_ramp_values(image, ramp)
        except Exception:
            snow = None

        if rain is None and snow is None:
            continue
        sample_source = rain if rain is not None else snow
        west, south, east, north = bounds
        x_pixel_size = (east - west) / max(sample_source.shape[1] - 1, 1)
        y_pixel_size = (south - north) / max(sample_source.shape[0] - 1, 1)
        if rain is not None:
            rain, latitudes, longitudes = sample_grid(rain, x_pixel_size, y_pixel_size, west, north)
        else:
            _, latitudes, longitudes = sample_grid(np.zeros_like(sample_source), x_pixel_size, y_pixel_size, west, north)
        if snow is not None:
            snow, _, _ = sample_grid(snow, x_pixel_size, y_pixel_size, west, north)
        frames.append(
            create_frame(
                provider_id=provider_id,
                name="ECCC GeoMet North American Radar",
                valid_time=ref_time,
                lead_minutes=offset,
                latitudes=latitudes,
                longitudes=longitudes,
                rain=rain,
                snow=snow,
                mixed=None,
                attribution="Environment and Climate Change Canada / MSC GeoMet",
                source_url=ECCC_RADAR_DOCS_URL,
                source_kind="wms-radar-and-extrapolation",
                metadata={"requested_urls": source_urls},
            )
        )

    if not frames:
        raise ValueError("ECCC GeoMet returned no usable rain or snow frames")
    return ProviderResult(
        id=provider_id,
        name="ECCC GeoMet North American Radar",
        status="ok",
        source_url=ECCC_RADAR_DOCS_URL,
        attribution="Environment and Climate Change Canada / MSC GeoMet",
        frames=frames,
        metadata={"anonymous": True, "forecast_offsets_minutes": FORECAST_OFFSETS_MINUTES},
    )


def parse_mrms_time_from_key(key: str) -> Optional[str]:
    match = re.search(r"MRMS_PrecipRate_00\.00_(\d{8})-(\d{6})", key)
    if not match:
        return None
    parsed = dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    return parsed.isoformat()


def list_mrms_keys(client: requests.Session, day: dt.date) -> List[str]:
    prefix = f"{NOAA_MRMS_PRODUCT_PREFIX}/{day:%Y%m%d}/"
    params = urlencode({"list-type": "2", "prefix": prefix})
    text, _ = get_text(client, f"{NOAA_MRMS_BUCKET_URL}/?{params}")
    root = ET.fromstring(text)
    keys = []
    for element in root.iter():
        if element.tag.endswith("Key") and element.text and element.text.endswith(".grib2.gz"):
            keys.append(element.text)
    return sorted(keys)


def latest_mrms_key(client: requests.Session) -> str:
    today = dt.datetime.now(dt.timezone.utc).date()
    for day in (today, today - dt.timedelta(days=1)):
        keys = list_mrms_keys(client, day)
        if keys:
            return keys[-1]
    raise ValueError("No current MRMS PrecipRate files found")


def fetch_noaa_mrms(client: requests.Session) -> ProviderResult:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("NOAA MRMS decoding requires xarray and cfgrib") from exc

    key = latest_mrms_key(client)
    url = f"{NOAA_MRMS_BUCKET_URL}/{key}"
    grib_gz_bytes, http_time = get_bytes(client, url)
    grib_bytes = gzip.decompress(grib_gz_bytes)
    with tempfile.NamedTemporaryFile(suffix=".grib2") as handle:
        handle.write(grib_bytes)
        handle.flush()
        dataset = xr.open_dataset(handle.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
        data = next(iter(dataset.data_vars.values()))
        values = data.values.astype(np.float32)
        latitudes = dataset["latitude"].values.astype(np.float64)
        longitudes = dataset["longitude"].values.astype(np.float64)
        valid_time = dataset.coords.get("valid_time")
        ref_time = None
        if valid_time is not None:
            ref_time = np.datetime_as_string(valid_time.values, unit="s") + "+00:00"

    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    values = np.clip(values, 0.0, 250.0).astype(np.float32)
    values, latitudes, longitudes = normalize_lonlat_arrays(values, latitudes, longitudes)
    frame = create_frame(
        provider_id="noaa_mrms",
        name="NOAA MRMS Precipitation Rate",
        valid_time=ref_time or parse_mrms_time_from_key(key) or http_time,
        lead_minutes=0,
        latitudes=latitudes,
        longitudes=longitudes,
        rain=values,
        snow=None,
        mixed=None,
        attribution="NOAA MRMS / AWS Open Data",
        source_url=NOAA_MRMS_DOCS_URL,
        source_kind="grib2-s3-open-data",
        metadata={"object_key": key, "phase_from_model": True},
    )
    return ProviderResult(
        id="noaa_mrms",
        name="NOAA MRMS Precipitation Rate",
        status="ok",
        source_url=NOAA_MRMS_DOCS_URL,
        attribution="NOAA MRMS / AWS Open Data",
        frames=[frame],
        metadata={"anonymous": True},
    )


def mrms_reflectivity_latest_url(product_path: str) -> str:
    product_name = product_path.split("/")[-1]
    return f"{NOAA_MRMS_NCEP_BASE_URL}/{product_path}/MRMS_{product_name}.latest.grib2.gz"


def parse_mrms_reflectivity_grib(grib_gz_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("NOAA MRMS reflectivity decoding requires xarray and cfgrib") from exc

    grib_bytes = gzip.decompress(grib_gz_bytes)
    with tempfile.NamedTemporaryFile(suffix=".grib2") as handle:
        handle.write(grib_bytes)
        handle.flush()
        dataset = xr.open_dataset(handle.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
        data = next(iter(dataset.data_vars.values()))
        dbz = data.values.astype(np.float32)
        latitudes = dataset["latitude"].values.astype(np.float64)
        longitudes = dataset["longitude"].values.astype(np.float64)
        valid_time = dataset.coords.get("valid_time")
        ref_time = None
        if valid_time is not None:
            ref_time = np.datetime_as_string(valid_time.values, unit="s") + "+00:00"

    values = dbz_to_mm_h(np.where(dbz < -900.0, -33.0, dbz))
    values, latitudes, longitudes = normalize_lonlat_arrays(values, latitudes, longitudes)
    return values, latitudes, longitudes, ref_time


def fetch_noaa_mrms_reflectivity_regions(client: requests.Session) -> ProviderResult:
    provider_id = "noaa_mrms_reflectivity"
    frames: List[PrecipFrame] = []
    errors = []
    for region_id, (product_path, _bounds) in MRMS_REFLECTIVITY_PRODUCTS.items():
        url = mrms_reflectivity_latest_url(product_path)
        try:
            grib_gz_bytes, http_time = get_bytes(client, url)
            values, latitudes, longitudes, ref_time = parse_mrms_reflectivity_grib(grib_gz_bytes)
            frames.append(
                create_frame(
                    provider_id=f"{provider_id}_{region_id.lower()}",
                    name=f"NOAA MRMS {region_id} Reflectivity",
                    valid_time=ref_time or http_time,
                    lead_minutes=0,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    rain=values,
                    snow=None,
                    mixed=None,
                    attribution="NOAA MRMS / NCEP",
                    source_url=NOAA_MRMS_DOCS_URL,
                    source_kind="grib2-radar-dbz",
                    metadata={
                        "region": region_id,
                        "product_path": product_path,
                        "requested_url": url,
                        "phase_from_model": True,
                        "native_parameter": "reflectivity_dbz",
                    },
                )
            )
        except Exception as exc:
            errors.append(f"{region_id}: {exc}")
    if not frames:
        raise ValueError("No usable NOAA MRMS reflectivity regions found. " + " | ".join(errors))
    return ProviderResult(
        id=provider_id,
        name="NOAA MRMS Regional Reflectivity",
        status="ok" if not errors else "partial",
        source_url=NOAA_MRMS_DOCS_URL,
        attribution="NOAA MRMS / NCEP",
        frames=frames,
        metadata={"anonymous": True, "phase_from_model": True, "warnings": errors},
    )


def fetch_iem_n0q(client: requests.Session) -> ProviderResult:
    provider_id = "iem_n0q"
    frames: List[PrecipFrame] = []
    errors = []
    for region_id, (live_dir, bounds) in IEM_REGIONS.items():
        url = f"{IEM_BASE_URL}/data/gis/images/4326/{live_dir}/n0q_0.png"
        try:
            image_bytes, ref_time = get_bytes(client, url)
            values = decode_iem_n0q_png(image_bytes)
            frames.append(
                create_model_phase_radar_frame(
                    f"{provider_id}_{region_id.lower()}",
                    f"IEM NEXRAD {region_id} Composite",
                    ref_time,
                    bounds,
                    values,
                    "Iowa Environmental Mesonet / NOAA NEXRAD",
                    IEM_BASE_URL,
                    "palette-indexed-png-radar-dbz",
                    {
                        "region": region_id,
                        "requested_url": url,
                        "phase_from_model": True,
                        "native_parameter": "reflectivity_dbz",
                    },
                )
            )
        except Exception as exc:
            errors.append(f"{region_id}: {exc}")
    if not frames:
        raise ValueError("No usable IEM N0Q radar regions found. " + " | ".join(errors))
    return ProviderResult(
        id=provider_id,
        name="IEM NEXRAD Regional Composites",
        status="ok" if not errors else "partial",
        source_url=IEM_BASE_URL,
        attribution="Iowa Environmental Mesonet / NOAA NEXRAD",
        frames=frames,
        metadata={"anonymous": True, "phase_from_model": True, "warnings": errors},
    )


def latest_gfs_run() -> Tuple[str, str]:
    now = dt.datetime.now(dt.timezone.utc)
    check_time = now - dt.timedelta(hours=4, minutes=30)
    possible_runs = [0, 6, 12, 18]
    run_hour = max([hour for hour in possible_runs if hour <= check_time.hour], default=18)
    if run_hour == 18 and check_time.hour < 4:
        check_time -= dt.timedelta(days=1)
    return check_time.strftime("%Y%m%d"), f"{run_hour:02d}"


def gfs_filter_url(date_text: str, run_hour: str, forecast_hour: int) -> str:
    f_str = f"{forecast_hour:03d}"
    params = {
        "file": f"gfs.t{run_hour}z.pgrb2.0p25.f{f_str}",
        "lev_2_m_above_ground": "on",
        "lev_surface": "on",
        "var_TMP": "on",
        "var_RH": "on",
        "var_PRATE": "on",
        "dir": f"/gfs.{date_text}/{run_hour}/atmos",
    }
    return f"{NOAA_GFS_FILTER_URL}?{urlencode(params)}"


def fetch_noaa_gfs_model(client: requests.Session, forecast_hour: int) -> PrecipFrame:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("NOAA GFS decoding requires xarray and cfgrib") from exc

    first_date, first_hour = latest_gfs_run()
    first_run = dt.datetime.strptime(f"{first_date}{first_hour}", "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    last_error: Optional[Exception] = None
    for offset in range(0, 5):
        run_time = first_run - dt.timedelta(hours=6 * offset)
        date_text = run_time.strftime("%Y%m%d")
        hour_text = run_time.strftime("%H")
        url = gfs_filter_url(date_text, hour_text, forecast_hour)
        try:
            grib_bytes, _ = get_bytes(client, url)
            with tempfile.NamedTemporaryFile(suffix=".grib2") as handle:
                handle.write(grib_bytes)
                handle.flush()
                dataset = xr.open_dataset(handle.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
                temp = data_var_by_grib_name(dataset, ["2t", "t2m", "tmp", "temperature"])
                rh = data_var_by_grib_name(dataset, ["2r", "r2", "rh", "relative humidity"])
                prate = data_var_by_grib_name(dataset, ["prate", "precipitation rate"])
                precip_rate = np.clip(prate.values.astype(np.float32) * 3600.0, 0.0, 250.0)
                wet_bulb = wet_bulb_temperature_c(temp.values.astype(np.float32), rh.values.astype(np.float32))
                valid_time = dataset.coords.get("valid_time")
                if valid_time is None:
                    valid_time = dataset.coords.get("time")
                ref_time = None
                if valid_time is not None:
                    ref_time = np.datetime_as_string(valid_time.values, unit="s") + "+00:00"
                latitudes = dataset["latitude"].values.astype(np.float64)
                longitudes = dataset["longitude"].values.astype(np.float64)
            precip_rate, latitudes, longitudes = normalize_lonlat_arrays(precip_rate, latitudes, longitudes)
            wet_bulb, _, _ = normalize_lonlat_arrays(wet_bulb, dataset["latitude"].values.astype(np.float64), dataset["longitude"].values.astype(np.float64))
            rain, snow, mixed = split_precip_by_wet_bulb(precip_rate, wet_bulb)
            return create_frame(
                provider_id="noaa_gfs",
                name="NOAA GFS Short-Range Precipitation Fallback",
                valid_time=ref_time or run_time.isoformat(),
                lead_minutes=forecast_hour * 60,
                latitudes=latitudes,
                longitudes=longitudes,
                rain=rain,
                snow=snow,
                mixed=mixed,
                attribution="NOAA GFS / NOMADS",
                source_url=url,
                source_kind="model-grib2",
                metadata={"forecast_hour": forecast_hour, "model_fallback": True},
            )
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(f"No usable GFS f{forecast_hour:03d} frame found: {last_error}")


def interpolate_gfs_frame(low: PrecipFrame, high: PrecipFrame, offset_minutes: int) -> PrecipFrame:
    span = max(1, high.lead_minutes - low.lead_minutes)
    alpha = min(1.0, max(0.0, (offset_minutes - low.lead_minutes) / span))
    rain = (1.0 - alpha) * low.rain_mm_h + alpha * high.rain_mm_h
    snow = None
    if low.snow_mm_h is not None and high.snow_mm_h is not None:
        snow = (1.0 - alpha) * low.snow_mm_h + alpha * high.snow_mm_h
    mixed = None
    if low.mixed_mm_h is not None and high.mixed_mm_h is not None:
        mixed = (1.0 - alpha) * low.mixed_mm_h + alpha * high.mixed_mm_h
    return create_frame(
        provider_id="noaa_gfs",
        name="NOAA GFS Short-Range Precipitation Fallback",
        valid_time=high.valid_time or low.valid_time,
        lead_minutes=offset_minutes,
        latitudes=low.latitudes,
        longitudes=low.longitudes,
        rain=rain,
        snow=snow,
        mixed=mixed,
        attribution="NOAA GFS / NOMADS",
        source_url=low.source_url,
        source_kind="model-grib2-interpolated",
        metadata={
            "forecast_offset_minutes": offset_minutes,
            "model_fallback": True,
            "interpolated_from_minutes": [low.lead_minutes, high.lead_minutes],
        },
    )


def select_or_interpolate_gfs_frame(frames: List[PrecipFrame], offset_minutes: int) -> PrecipFrame:
    sorted_frames = sorted(frames, key=lambda frame: frame.lead_minutes)
    for frame in sorted_frames:
        if frame.lead_minutes == offset_minutes:
            return frame

    lower = [frame for frame in sorted_frames if frame.lead_minutes < offset_minutes]
    upper = [frame for frame in sorted_frames if frame.lead_minutes > offset_minutes]
    if lower and upper:
        return interpolate_gfs_frame(lower[-1], upper[0], offset_minutes)

    source = lower[-1] if lower else sorted_frames[0]
    return create_frame(
        provider_id="noaa_gfs",
        name="NOAA GFS Short-Range Precipitation Fallback",
        valid_time=source.valid_time,
        lead_minutes=offset_minutes,
        latitudes=source.latitudes,
        longitudes=source.longitudes,
        rain=source.rain_mm_h,
        snow=source.snow_mm_h,
        mixed=source.mixed_mm_h,
        attribution="NOAA GFS / NOMADS",
        source_url=source.source_url,
        source_kind="model-grib2-persistence",
        metadata={
            "forecast_offset_minutes": offset_minutes,
            "model_fallback": True,
            "persisted_from_minutes": source.lead_minutes,
        },
    )


def fetch_noaa_gfs_provider(client: requests.Session) -> ProviderResult:
    frames = []
    errors = []
    for forecast_hour in NOAA_GFS_FORECAST_HOURS:
        try:
            frames.append(fetch_noaa_gfs_model(client, forecast_hour))
        except Exception as exc:
            errors.append(f"f{forecast_hour:03d}: {exc}")
    if not frames:
        raise ValueError("No usable GFS model frames found. " + " | ".join(errors))

    interpolated: List[PrecipFrame] = []
    for offset in FORECAST_OFFSETS_MINUTES:
        frame = select_or_interpolate_gfs_frame(frames, offset)
        frame.metadata["model_fallback"] = True
        frame.metadata["display_offset_minutes"] = offset
        interpolated.append(frame)
    return ProviderResult(
        id="noaa_gfs",
        name="NOAA GFS Short-Range Precipitation Fallback",
        status="ok",
        source_url="https://nomads.ncep.noaa.gov/",
        attribution="NOAA GFS / NOMADS",
        frames=interpolated,
        metadata={
            "anonymous": True,
            "model_fallback": True,
            "forecast_hours_requested": NOAA_GFS_FORECAST_HOURS,
            "warnings": errors,
        },
    )


def latest_dwd_member(client: requests.Session, base_url: str, pattern: str) -> Tuple[str, Optional[str]]:
    text, _ = get_text(client, base_url)
    matches = re.findall(r'href="([^"]+)"', text)
    candidates = [name for name in matches if re.match(pattern, name)]
    if not candidates:
        raise ValueError(f"No DWD files matching {pattern}")
    filename = sorted(candidates)[-1]
    return base_url + filename, filename


def decode_radolan_best_effort(payload: bytes) -> np.ndarray:
    marker = payload.find(b"\x03")
    if marker < 0:
        raise ValueError("Could not locate RADOLAN header terminator")
    data = payload[marker + 1 :]
    arr = np.frombuffer(data, dtype="<u2")
    possible_shapes = [(900, 900), (1100, 900), (1200, 1100), (1500, 1400)]
    for rows, cols in possible_shapes:
        if arr.size >= rows * cols:
            raw_grid = arr[: rows * cols].reshape(rows, cols)
            grid = np.where((raw_grid & 0x1000) > 0, 0.0, raw_grid & 0x0FFF)
            return np.clip(grid / 10.0, 0.0, 250.0).astype(np.float32)
    raise ValueError(f"Unsupported RADOLAN payload size: {arr.size}")


def fetch_dwd_open_data(client: requests.Session) -> ProviderResult:
    frames: List[PrecipFrame] = []
    errors = []
    for kind, base_url, pattern in (
        ("rain", DWD_RADAR_RV_URL, r".*[Rr][Vv].*(?:\.tar\.bz2|\.gz|bin)$"),
        ("snow", DWD_RADAR_RS_URL, r".*[Rr][Ss].*(?:\.tar\.bz2|\.gz|bin)$"),
    ):
        try:
            url, filename = latest_dwd_member(client, base_url, pattern)
            payload, ref_time = get_bytes(client, url)
            if filename.endswith(".tar.bz2"):
                with tarfile.open(fileobj=io.BytesIO(bz2.decompress(payload))) as archive:
                    member = next((item for item in archive.getmembers() if item.isfile()), None)
                    if member is None:
                        raise ValueError("DWD archive contains no file member")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError("DWD archive member could not be read")
                    payload = extracted.read()
            elif filename.endswith(".gz"):
                payload = gzip.decompress(payload)
            values = decode_radolan_best_effort(payload)
            bounds = REGION_BOUNDS["Germany"]
            west, south, east, north = bounds
            x_pixel_size = (east - west) / max(values.shape[1] - 1, 1)
            y_pixel_size = (south - north) / max(values.shape[0] - 1, 1)
            values, latitudes, longitudes = sample_grid(values, x_pixel_size, y_pixel_size, west, north)
            frames.append(
                create_frame(
                    provider_id="dwd_open_data",
                    name="DWD Open Data Radar Composite",
                    valid_time=ref_time,
                    lead_minutes=0,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    rain=values if kind == "rain" else None,
                    snow=values if kind == "snow" else None,
                    mixed=None,
                    attribution="Deutscher Wetterdienst / DWD Open Data",
                    source_url=DWD_SOURCE_URL,
                    source_kind="radolan-tar-bz2",
                    metadata={"file": filename, "kind": kind},
                )
            )
        except Exception as exc:
            errors.append(f"{kind}: {exc}")
    if not frames:
        raise ValueError("; ".join(errors) if errors else "No DWD frames decoded")
    return ProviderResult(
        id="dwd_open_data",
        name="DWD Open Data Radar Composite",
        status="ok" if not errors else "partial",
        source_url=DWD_SOURCE_URL,
        attribution="Deutscher Wetterdienst / DWD Open Data",
        frames=frames,
        metadata={"anonymous": True, "warnings": errors},
    )


def fetch_fmi_open_data(client: requests.Session) -> ProviderResult:
    provider_id = "fmi_open_data"
    bounds = REGION_BOUNDS["Finland"]
    width = int(os.getenv("FMI_RADAR_WIDTH", "1100") or "1100")
    height = int(os.getenv("FMI_RADAR_HEIGHT", "1200") or "1200")
    url = wms_get_map_url(
        FMI_WMS_URL,
        FMI_RAIN_RATE_LAYER,
        bounds,
        width=width,
        height=height,
    )
    image_bytes, ref_time = get_bytes(client, url)
    with Image.open(io.BytesIO(image_bytes)) as image:
        values = rgba_to_color_ramp_values(image, FMI_RAIN_RAMP, alpha_threshold=64)

    west, south, east, north = bounds
    x_pixel_size = (east - west) / max(values.shape[1] - 1, 1)
    y_pixel_size = (south - north) / max(values.shape[0] - 1, 1)
    values, latitudes, longitudes = sample_grid(values, x_pixel_size, y_pixel_size, west, north)
    frame = create_frame(
        provider_id=provider_id,
        name="FMI Open Data Finland Radar",
        valid_time=ref_time,
        lead_minutes=0,
        latitudes=latitudes,
        longitudes=longitudes,
        rain=values,
        snow=None,
        mixed=None,
        attribution="Finnish Meteorological Institute / FMI Open Data",
        source_url=FMI_OPEN_DATA_URL,
        source_kind="wms-radar-rain-rate",
        metadata={
            "requested_url": url,
            "phase_from_model": True,
            "license": "CC BY 4.0",
            "license_url": FMI_LICENSE_URL,
            "wms_layer": FMI_RAIN_RATE_LAYER,
        },
    )
    return ProviderResult(
        id=provider_id,
        name="FMI Open Data Finland Radar",
        status="ok",
        source_url=FMI_OPEN_DATA_URL,
        attribution="Finnish Meteorological Institute / FMI Open Data",
        frames=[frame],
        metadata={
            "anonymous": True,
            "license": "CC BY 4.0",
            "license_url": FMI_LICENSE_URL,
            "wms_layer": FMI_RAIN_RATE_LAYER,
        },
    )


def region_axes_from_bounds_and_shape(bounds: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
    west, south, east, north = bounds
    rows, cols = shape
    x_pixel_size = (east - west) / max(cols - 1, 1)
    y_pixel_size = (south - north) / max(rows - 1, 1)
    return x_pixel_size, y_pixel_size, west, north


def create_model_phase_radar_frame(
    provider_id: str,
    name: str,
    valid_time: Optional[str],
    bounds: Tuple[float, float, float, float],
    values: np.ndarray,
    attribution: str,
    source_url: str,
    source_kind: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> PrecipFrame:
    x_pixel_size, y_pixel_size, west, north = region_axes_from_bounds_and_shape(bounds, values.shape)
    sampled, latitudes, longitudes = sample_grid(values, x_pixel_size, y_pixel_size, west, north)
    frame_metadata = {"phase_from_model": True}
    frame_metadata.update(metadata or {})
    return create_frame(
        provider_id=provider_id,
        name=name,
        valid_time=valid_time,
        lead_minutes=0,
        latitudes=latitudes,
        longitudes=longitudes,
        rain=sampled,
        snow=None,
        mixed=None,
        attribution=attribution,
        source_url=source_url,
        source_kind=source_kind,
        metadata=frame_metadata,
    )


def latest_marn_entry(client: requests.Session) -> Tuple[dt.datetime, str]:
    now = dt.datetime.now(dt.timezone.utc)
    local_now = now - dt.timedelta(hours=6)
    entries: List[Tuple[dt.datetime, str]] = []
    for days_back in range(0, 2):
        date_text = (local_now - dt.timedelta(days=days_back)).strftime("%Y-%m-%d")
        prefix = f"{MARN_PRODUCT_PREFIX}{date_text}"
        text, _ = get_text(client, f"{MARN_BUCKET_API_URL}?{urlencode({'prefix': prefix, 'maxResults': '500'})}")
        payload = json.loads(text)
        for item in payload.get("items", []):
            name = item.get("name", "")
            stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            try:
                local = dt.datetime.strptime(stem, "%Y-%m-%d %H-%M-%S")
            except ValueError:
                continue
            parsed = local.replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=6)
            if parsed.year >= 2000:
                entries.append((parsed, name))
    if not entries:
        raise ValueError("No MARN radar images found")
    return sorted(entries, key=lambda item: item[0])[-1]


def fetch_marn_el_salvador(client: requests.Session) -> ProviderResult:
    provider_id = "marn_el_salvador"
    valid_dt, object_name = latest_marn_entry(client)
    url = f"{MARN_BUCKET_URL}/radar-images-sv/{quote(object_name)}"
    image_bytes, ref_time = get_bytes(client, url)
    values = decode_marn_png(image_bytes)
    frame = create_model_phase_radar_frame(
        provider_id,
        "MARN/SNET El Salvador Radar",
        valid_dt.isoformat() if valid_dt else ref_time,
        REGION_BOUNDS["El Salvador"],
        values,
        "MARN El Salvador / SNET",
        "https://storage.googleapis.com/radar-images-sv",
        "gcs-radar-png-dbz",
        {"object_name": object_name},
    )
    return ProviderResult(
        id=provider_id,
        name="MARN/SNET El Salvador Radar",
        status="ok",
        source_url="https://storage.googleapis.com/radar-images-sv",
        attribution="MARN El Salvador / SNET",
        frames=[frame],
        metadata={"anonymous": True, "phase_from_model": True},
    )


def cwa_url_for_timestamp(timestamp: dt.datetime) -> str:
    rounded_minute = (timestamp.minute // 10) * 10
    rounded = timestamp.replace(minute=rounded_minute, second=0, microsecond=0)
    local = rounded.astimezone(dt.timezone.utc) + dt.timedelta(hours=8)
    filename = local.strftime("%Y%m%d%H%M") + "compref_mosaic.xml"
    return f"{CWA_BASE_URL}{CWA_ARCHIVE_PREFIX}/{filename}"


def fetch_cwa_taiwan(client: requests.Session) -> ProviderResult:
    provider_id = "cwa_taiwan"
    now = dt.datetime.now(dt.timezone.utc)
    errors = []
    for step in range(0, 5):
        target = now - dt.timedelta(minutes=10 * step)
        url = cwa_url_for_timestamp(target)
        try:
            xml_bytes, ref_time = get_bytes(client, url)
            values = parse_cwa_xml(xml_bytes)
            frame = create_model_phase_radar_frame(
                provider_id,
                "CWA Taiwan QPESUMS Radar",
                ref_time,
                REGION_BOUNDS["Taiwan"],
                values,
                "Taiwan Central Weather Administration / Open Data",
                "https://opendata.cwa.gov.tw/",
                "s3-radar-xml-dbz",
                {"requested_url": url},
            )
            return ProviderResult(
                id=provider_id,
                name="CWA Taiwan QPESUMS Radar",
                status="ok",
                source_url="https://opendata.cwa.gov.tw/",
                attribution="Taiwan Central Weather Administration / Open Data",
                frames=[frame],
                metadata={"anonymous": True, "phase_from_model": True},
            )
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("No usable CWA Taiwan radar XML found: " + " | ".join(errors[-2:]))


def fetch_met_malaysia(client: requests.Session) -> ProviderResult:
    provider_id = "met_malaysia"
    image_bytes, ref_time = get_bytes(client, MMD_RADAR_GIF_URL)
    with Image.open(io.BytesIO(image_bytes)) as image:
        frames = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise ValueError("MET Malaysia GIF contained no frames")
    latest = frames[-1]
    region_specs = [
        ("MYPENINSULAR", "Malaysia Peninsular"),
        ("MYEAST", "Malaysia East"),
    ]
    precip_frames: List[PrecipFrame] = []
    for region_name, bounds_key in region_specs:
        y0, y1, x0, x1 = MMD_SUBRECTS[region_name]
        values = fill_thin_gaps(decode_mmd_rgb(latest[y0:y1, x0:x1, :]))
        precip_frames.append(
            create_model_phase_radar_frame(
                f"{provider_id}_{region_name.lower()}",
                "MET Malaysia Radar Composite",
                ref_time,
                REGION_BOUNDS[bounds_key],
                values,
                "MET Malaysia / api.met.gov.my",
                "https://api.met.gov.my/",
                "animated-gif-radar-dbz",
                {"region": region_name, "phase_from_model": True},
            )
        )
    return ProviderResult(
        id=provider_id,
        name="MET Malaysia Radar Composite",
        status="ok",
        source_url="https://api.met.gov.my/",
        attribution="MET Malaysia / api.met.gov.my",
        frames=precip_frames,
        metadata={"anonymous": True, "phase_from_model": True, "frames_in_gif": len(frames)},
    )


def opera_url_for_timestamp(timestamp: dt.datetime) -> str:
    rounded_minute = (timestamp.minute // 5) * 5
    rounded = timestamp.replace(minute=rounded_minute, second=0, microsecond=0)
    filename = rounded.strftime("OPERA@%Y%m%dT%H%M@0@DBZH.h5")
    path = rounded.strftime(f"%Y/%m/%d/OPERA/COMP/{filename}")
    return f"{OPERA_BASE_URL}/{path}"


def parse_opera_hdf5(payload: bytes) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("OPERA HDF5 decoding requires h5py") from exc
    with h5py.File(io.BytesIO(payload), "r") as handle:
        raw = handle["dataset1/data1/data"][:]
        what = handle["dataset1/data1/what"]
        nodata = float(what.attrs["nodata"])
        undetect = float(what.attrs["undetect"])
        gain = float(what.attrs["gain"])
        offset = float(what.attrs["offset"])
    dbz = raw.astype(np.float32) * gain + offset
    invalid = np.isclose(raw, nodata, atol=1.0) | np.isclose(raw, undetect, atol=1.0)
    return dbz_to_mm_h(np.where(invalid, -33.0, dbz))


def fetch_opera_europe(client: requests.Session) -> ProviderResult:
    provider_id = "opera_europe"
    now = dt.datetime.now(dt.timezone.utc)
    errors = []
    for step in range(0, 5):
        target = now - dt.timedelta(minutes=5 * step)
        url = opera_url_for_timestamp(target)
        try:
            payload, ref_time = get_bytes(client, url)
            values = parse_opera_hdf5(payload)
            frame = create_model_phase_radar_frame(
                provider_id,
                "EUMETNET OPERA Europe Radar",
                ref_time,
                REGION_BOUNDS["Europe OPERA"],
                values,
                "EUMETNET OPERA",
                "https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/",
                "hdf5-radar-dbz",
                {"requested_url": url, "phase_from_model": True},
            )
            return ProviderResult(
                id=provider_id,
                name="EUMETNET OPERA Europe Radar",
                status="ok",
                source_url="https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/",
                attribution="EUMETNET OPERA",
                frames=[frame],
                metadata={"anonymous": True, "phase_from_model": True},
            )
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("No usable OPERA Europe HDF5 found: " + " | ".join(errors[-2:]))


def latest_dpc_timestamp_ms(client: requests.Session) -> int:
    response = client.get(f"{DPC_API_BASE_URL}/findLastProductByType?type=VMI", timeout=TIMEOUT)
    response.raise_for_status()
    products = response.json().get("lastProducts") or []
    if not products:
        raise ValueError("DPC findLastProductByType returned no products")
    return int(products[0]["time"])


def dpc_download_url(client: requests.Session, timestamp_ms: int) -> str:
    response = client.post(
        f"{DPC_API_BASE_URL}/downloadProduct",
        json={"productType": "VMI", "productDate": timestamp_ms},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if "url" not in payload:
        raise ValueError("DPC downloadProduct did not return a URL")
    return payload["url"]


def parse_dpc_tiff(payload: bytes) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("DPC GeoTIFF decoding requires tifffile") from exc
    values = tifffile.imread(io.BytesIO(payload)).astype(np.float32)
    if values.ndim != 2:
        raise ValueError(f"DPC GeoTIFF has unexpected shape: {values.shape}")
    return dbz_to_mm_h(np.where(values < -100.0, -33.0, values))


def fetch_dpc_italy(client: requests.Session) -> ProviderResult:
    provider_id = "dpc_italy"
    latest_ms = latest_dpc_timestamp_ms(client)
    errors = []
    for step in range(0, 4):
        timestamp_ms = latest_ms - step * 5 * 60 * 1000
        try:
            url = dpc_download_url(client, timestamp_ms)
            payload, ref_time = get_bytes(client, url)
            values = parse_dpc_tiff(payload)
            frame = create_model_phase_radar_frame(
                provider_id,
                "Radar-DPC Italy VMI Radar",
                dt.datetime.fromtimestamp(timestamp_ms / 1000.0, tz=dt.timezone.utc).isoformat(),
                REGION_BOUNDS["Italy DPC"],
                values,
                "Radar-DPC / Dipartimento della Protezione Civile",
                "https://radar-api.protezionecivile.it",
                "geotiff-radar-dbz",
                {"requested_url": url, "http_ref_time": ref_time, "phase_from_model": True},
            )
            return ProviderResult(
                id=provider_id,
                name="Radar-DPC Italy VMI Radar",
                status="ok",
                source_url="https://radar-api.protezionecivile.it",
                attribution="Radar-DPC / Dipartimento della Protezione Civile",
                frames=[frame],
                metadata={"anonymous": True, "phase_from_model": True, "license": "CC-BY-SA 4.0"},
            )
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("No usable DPC Italy GeoTIFF found: " + " | ".join(errors[-2:]))


def lon_to_tile_x(lon: float, zoom: int) -> int:
    return int(math.floor((lon + 180.0) / 360.0 * (2**zoom)))


def lat_to_tile_y(lat: float, zoom: int) -> int:
    lat_rad = math.radians(lat)
    return int(math.floor((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * (2**zoom)))


def tile_pixel_to_lonlat(x: int, y: int, zoom: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
    n = float((2**zoom) * size)
    cols = np.arange(x * size, (x + 1) * size, dtype=np.float64) + 0.5
    rows = np.arange(y * size, (y + 1) * size, dtype=np.float64) + 0.5
    longitudes = cols / n * 360.0 - 180.0
    mercator = math.pi * (1.0 - 2.0 * rows / n)
    latitudes = np.degrees(np.arctan(np.sinh(mercator)))
    return longitudes, latitudes


def parse_jma_time(value: str) -> str:
    return dt.datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc).isoformat()


def fetch_jma_nowcast(client: requests.Session) -> ProviderResult:
    response = client.get(JMA_NOWCAST_TIMES_URL, timeout=TIMEOUT)
    response.raise_for_status()
    times = response.json()
    if not times:
        raise ValueError("JMA nowcast time feed is empty")

    zoom = int(os.getenv("JMA_RADAR_ZOOM", "6") or "6")
    bounds = REGION_BOUNDS["Japan"]
    west, south, east, north = bounds
    x_start = lon_to_tile_x(west, zoom)
    x_end = lon_to_tile_x(east, zoom)
    y_start = lat_to_tile_y(north, zoom)
    y_end = lat_to_tile_y(south, zoom)
    frames = []

    for frame_info in times[:1]:
        base_time = frame_info["basetime"]
        valid_time = frame_info["validtime"]
        rows = []
        for tile_y in range(y_start, y_end + 1):
            row_tiles = []
            for tile_x in range(x_start, x_end + 1):
                url = JMA_NOWCAST_TILE_TEMPLATE.format(
                    base_time=base_time,
                    valid_time=valid_time,
                    z=zoom,
                    x=tile_x,
                    y=tile_y,
                )
                image_bytes, _ = get_bytes(client, url)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    row_tiles.append(rgba_to_configured_values(image, JMA_HRPNS_COLOR_VALUES))
            rows.append(np.concatenate(row_tiles, axis=1))
        values = np.concatenate(rows, axis=0)
        lon_values = [tile_pixel_to_lonlat(tile_x, y_start, zoom, 256)[0] for tile_x in range(x_start, x_end + 1)]
        longitudes = np.concatenate(lon_values)
        lat_values = [tile_pixel_to_lonlat(x_start, tile_y, zoom, 256)[1] for tile_y in range(y_start, y_end + 1)]
        latitudes = np.concatenate(lat_values)
        lon_mask = (longitudes >= west) & (longitudes <= east)
        lat_mask = (latitudes >= south) & (latitudes <= north)
        values = values[np.ix_(lat_mask, lon_mask)]
        latitudes = latitudes[lat_mask]
        longitudes = longitudes[lon_mask]
        values, latitudes, longitudes = normalize_lonlat_arrays(values, latitudes, longitudes)
        frames.append(
            create_frame(
                provider_id="jma_nowcast",
                name="JMA Nowcast Radar",
                valid_time=parse_jma_time(valid_time),
                lead_minutes=0,
                latitudes=latitudes,
                longitudes=longitudes,
                rain=values,
                snow=None,
                mixed=None,
                attribution="Japan Meteorological Agency",
                source_url="https://www.jma.go.jp/jp/radnowc/index.html",
                source_kind="xyz-raster-tiles",
                metadata={"base_time": base_time, "valid_time": valid_time, "phase_from_model": True},
            )
        )
    return ProviderResult(
        id="jma_nowcast",
        name="JMA Nowcast Radar",
        status="ok",
        source_url="https://www.jma.go.jp/jp/radnowc/index.html",
        attribution="Japan Meteorological Agency",
        frames=frames,
        metadata={"anonymous": True},
    )


def parse_bounds(value: Any) -> Tuple[float, float, float, float]:
    if isinstance(value, str):
        parts = [float(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("bounds must be west,south,east,north")
        return parts[0], parts[1], parts[2], parts[3]
    if isinstance(value, list) and len(value) == 4:
        return float(value[0]), float(value[1]), float(value[2]), float(value[3])
    raise ValueError("bounds must be west,south,east,north or a four-item list")


def configured_wms_url(spec: Dict[str, Any], bounds: Tuple[float, float, float, float]) -> str:
    wms = spec.get("wms") or {}
    service_url = wms.get("url") or spec.get("wms_url")
    layers = wms.get("layers") or spec.get("wms_layers")
    if not service_url or not layers:
        raise ValueError("WMS configured providers require wms.url and wms.layers")
    version = str(wms.get("version") or spec.get("wms_version") or "1.3.0")
    crs_key = "SRS" if version.startswith("1.1") else "CRS"
    crs = wms.get("crs") or spec.get("wms_crs") or "EPSG:4326"
    west, south, east, north = bounds
    bbox = f"{west},{south},{east},{north}" if crs_key == "SRS" else f"{south},{west},{north},{east}"
    params = {
        "SERVICE": "WMS",
        "VERSION": version,
        "REQUEST": "GetMap",
        "LAYERS": layers,
        "STYLES": wms.get("styles") or spec.get("wms_styles") or "",
        "FORMAT": wms.get("format") or spec.get("wms_format") or "image/png",
        "TRANSPARENT": str(wms.get("transparent", True)).lower(),
        crs_key: crs,
        "WIDTH": str(wms.get("width") or spec.get("width") or 1200),
        "HEIGHT": str(wms.get("height") or spec.get("height") or 1200),
        "BBOX": bbox,
    }
    if wms.get("time") or spec.get("time"):
        params["TIME"] = wms.get("time") or spec.get("time")
    return f"{service_url}?{urlencode(params)}"


def interpolate_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            return os.getenv(match.group(1), "")

        return re.sub(r"\$\{([A-Z0-9_]+)\}", replace, value)
    if isinstance(value, list):
        return [interpolate_env(item) for item in value]
    if isinstance(value, dict):
        return {key: interpolate_env(item) for key, item in value.items()}
    return value


def configured_get_bytes(client: requests.Session, url: str, spec: Dict[str, Any]) -> Tuple[bytes, Optional[str]]:
    headers = spec.get("headers") or {}
    response = client.get(url, headers=headers, timeout=TIMEOUT)
    response.raise_for_status()
    return response.content, parse_last_modified(response.headers)


def configured_values(image_bytes: bytes, spec: Dict[str, Any]) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        if "color_values" in spec:
            return rgba_to_configured_values(image, spec["color_values"])
        if "rain_color_values" in spec:
            return rgba_to_configured_values(image, spec["rain_color_values"])
    raise ValueError("Configured providers require color_values or rain_color_values")


def fetch_configured_providers(client: requests.Session) -> List[ProviderResult]:
    specs = load_configured_provider_specs(client)
    results = []
    for spec in specs:
        spec = interpolate_env(spec)
        provider_id = str(spec["id"])
        name = spec.get("name", provider_id)
        attribution = spec.get("attribution", provider_id)
        source_url = spec.get("source_url") or spec.get("image_url") or spec.get("wms_url") or RADAR_SOURCE_REFERENCE
        metadata = configured_provider_metadata(spec)
        if not configured_provider_enabled(spec):
            results.append(
                ProviderResult(
                    id=provider_id,
                    name=name,
                    status="skipped",
                    source_url=source_url,
                    attribution=attribution,
                    error="Configured provider is not enabled",
                    metadata=metadata,
                )
            )
            continue
        if not configured_provider_redistribution_allowed(spec):
            results.append(
                ProviderResult(
                    id=provider_id,
                    name=name,
                    status="blocked",
                    source_url=source_url,
                    attribution=attribution,
                    error="Redistribution is not explicitly allowed for this provider",
                    metadata=metadata,
                )
            )
            continue
        try:
            bounds = parse_bounds(spec.get("bounds"))
            image_url = spec.get("image_url") or configured_wms_url(spec, bounds)
            image_bytes, ref_time = configured_get_bytes(client, image_url, spec)
            values = configured_values(image_bytes, spec)
            west, south, east, north = bounds
            x_pixel_size = (east - west) / max(values.shape[1] - 1, 1)
            y_pixel_size = (south - north) / max(values.shape[0] - 1, 1)
            values, latitudes, longitudes = sample_grid(values, x_pixel_size, y_pixel_size, west, north)
            precip_kind = spec.get("precip_type", "rain")
            frame = create_frame(
                provider_id=provider_id,
                name=name,
                valid_time=spec.get("ref_time") or ref_time,
                lead_minutes=int(spec.get("lead_minutes", 0)),
                latitudes=latitudes,
                longitudes=longitudes,
                rain=values if precip_kind in ("rain", "all") else None,
                snow=values if precip_kind == "snow" else None,
                mixed=values if precip_kind == "mixed" else None,
                attribution=attribution,
                source_url=source_url,
                source_kind="configured-image-or-wms",
                metadata=metadata,
            )
            results.append(
                ProviderResult(
                    id=provider_id,
                    name=name,
                    status="ok",
                    source_url=source_url,
                    attribution=attribution,
                    frames=[frame],
                    metadata=metadata,
                )
            )
        except Exception as exc:
            results.append(
                ProviderResult(
                    id=provider_id,
                    name=name,
                    status="error",
                    source_url=source_url,
                    attribution=attribution,
                    error=str(exc),
                    metadata=metadata,
                )
            )
    return results


def provider_specs_from_payload(payload: Any, default_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    specs = payload.get("providers", payload) if isinstance(payload, dict) else payload
    if not specs:
        return []
    if not isinstance(specs, list):
        raise ValueError("Configured radar provider registry must be a list or an object with a providers list")
    output = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each configured radar provider must be an object")
        merged = dict(spec)
        if default_metadata:
            merged.setdefault("_registry", {}).update(default_metadata)
        output.append(merged)
    return output


def load_configured_provider_specs(client: requests.Session) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    registry_file = RADAR_PROVIDER_REGISTRY_FILE.strip()
    if registry_file and os.path.exists(registry_file):
        with open(registry_file, "r", encoding="utf-8") as handle:
            specs.extend(provider_specs_from_payload(json.load(handle), {"source": registry_file, "strict": True}))

    if RADAR_PROVIDER_REGISTRY_URL:
        text, _ = get_text(client, RADAR_PROVIDER_REGISTRY_URL)
        specs.extend(provider_specs_from_payload(json.loads(text), {"source": RADAR_PROVIDER_REGISTRY_URL, "strict": True}))

    raw = os.getenv("RADAR_PROVIDERS_JSON", "").strip()
    if raw:
        specs.extend(provider_specs_from_payload(json.loads(raw), {"source": "RADAR_PROVIDERS_JSON", "strict": False}))

    return specs


def configured_provider_enabled(spec: Dict[str, Any]) -> bool:
    if RADAR_ALLOW_UNVERIFIED_PROVIDERS:
        return True
    if "enabled" in spec:
        return bool(spec["enabled"])
    status = str(spec.get("status", "")).lower()
    if status:
        return status in {"enabled", "active", "ok"}
    registry = spec.get("_registry") or {}
    return not bool(registry.get("strict"))


def configured_provider_redistribution_allowed(spec: Dict[str, Any]) -> bool:
    if RADAR_ALLOW_UNVERIFIED_PROVIDERS:
        return True
    redistribution = spec.get("redistribution")
    if isinstance(redistribution, dict):
        return bool(redistribution.get("allowed"))
    if "redistribution_allowed" in spec:
        return bool(spec["redistribution_allowed"])
    registry = spec.get("_registry") or {}
    return not bool(registry.get("strict"))


def configured_provider_metadata(spec: Dict[str, Any]) -> Dict[str, Any]:
    redistribution = spec.get("redistribution") if isinstance(spec.get("redistribution"), dict) else {}
    return {
        "configured": True,
        "registry": spec.get("_registry") or {},
        "license": spec.get("license") or redistribution.get("license"),
        "license_url": spec.get("license_url") or redistribution.get("license_url"),
        "redistribution_allowed": configured_provider_redistribution_allowed(spec),
        "unverified_override_active": RADAR_ALLOW_UNVERIFIED_PROVIDERS,
        "provider_status": spec.get("status"),
        "notes": spec.get("notes"),
    }


def provider_error(provider_id: str, name: str, source_url: str, attribution: str, exc: Exception) -> ProviderResult:
    return ProviderResult(
        id=provider_id,
        name=name,
        status="error",
        source_url=source_url,
        attribution=attribution,
        error=str(exc),
    )


def fetch_all_providers(client: requests.Session) -> List[ProviderResult]:
    providers = [
        ("eccc_geomet", "ECCC GeoMet North American Radar", ECCC_RADAR_DOCS_URL, "Environment and Climate Change Canada / MSC GeoMet", fetch_eccc_geomet),
        ("noaa_mrms", "NOAA MRMS Precipitation Rate", NOAA_MRMS_DOCS_URL, "NOAA MRMS / AWS Open Data", fetch_noaa_mrms),
        ("noaa_mrms_reflectivity", "NOAA MRMS Regional Reflectivity", NOAA_MRMS_DOCS_URL, "NOAA MRMS / NCEP", fetch_noaa_mrms_reflectivity_regions),
        ("iem_n0q", "IEM NEXRAD Regional Composites", IEM_BASE_URL, "Iowa Environmental Mesonet / NOAA NEXRAD", fetch_iem_n0q),
        ("noaa_gfs", "NOAA GFS Short-Range Precipitation Fallback", "https://nomads.ncep.noaa.gov/", "NOAA GFS / NOMADS", fetch_noaa_gfs_provider),
        ("dwd_open_data", "DWD Open Data Radar Composite", DWD_SOURCE_URL, "Deutscher Wetterdienst / DWD Open Data", fetch_dwd_open_data),
        ("fmi_open_data", "FMI Open Data Finland Radar", FMI_OPEN_DATA_URL, "Finnish Meteorological Institute / FMI Open Data", fetch_fmi_open_data),
        ("jma_nowcast", "JMA Nowcast Radar", "https://www.jma.go.jp/jp/radnowc/index.html", "Japan Meteorological Agency", fetch_jma_nowcast),
        ("marn_el_salvador", "MARN/SNET El Salvador Radar", "https://storage.googleapis.com/radar-images-sv", "MARN El Salvador / SNET", fetch_marn_el_salvador),
        ("cwa_taiwan", "CWA Taiwan QPESUMS Radar", "https://opendata.cwa.gov.tw/", "Taiwan Central Weather Administration / Open Data", fetch_cwa_taiwan),
        ("met_malaysia", "MET Malaysia Radar Composite", "https://api.met.gov.my/", "MET Malaysia / api.met.gov.my", fetch_met_malaysia),
        ("opera_europe", "EUMETNET OPERA Europe Radar", "https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/", "EUMETNET OPERA", fetch_opera_europe),
        ("dpc_italy", "Radar-DPC Italy VMI Radar", "https://radar-api.protezionecivile.it", "Radar-DPC / Dipartimento della Protezione Civile", fetch_dpc_italy),
    ]
    results: List[ProviderResult] = []
    for provider_id, name, source_url, attribution, fetcher in providers:
        try:
            results.append(fetcher(client))
        except Exception as exc:
            results.append(provider_error(provider_id, name, source_url, attribution, exc))
    results.extend(fetch_configured_providers(client))
    return results


def frame_is_model(frame: PrecipFrame) -> bool:
    return bool(frame.metadata.get("model_fallback"))


def tile_has_model_permission(tile_bounds: Tuple[float, float, float, float], observation_bounds: List[Tuple[float, float, float, float]]) -> bool:
    if ENABLE_GLOBAL_MODEL_FALLBACK:
        return True
    return any(intersects_bounds(tile_bounds, bounds) for bounds in observation_bounds)


def make_work_grid(
    selected: List[Tuple[PrecipFrame, float]],
    model_frames: List[PrecipFrame],
    offset_minutes: int,
    tile_bounds: Tuple[float, float, float, float],
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    observation_bounds: List[Tuple[float, float, float, float]],
) -> WorkGrid:
    rain = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float32)
    snow = np.zeros_like(rain)
    mixed = np.zeros_like(rain)
    provider_ids = set()
    attributions = set()
    source_urls = set()
    ref_times: List[str] = []

    phase_model = None
    exact_phase_models = [frame for frame in model_frames if frame.lead_minutes == offset_minutes and intersects_bounds(tile_bounds, frame.bounds)]
    if exact_phase_models:
        phase_model = exact_phase_models[0]
    model_phase_rain = model_phase_snow = model_phase_mixed = None
    if phase_model is not None:
        model_phase_rain = resample_field(phase_model, "rain_mm_h", lat_vals, lon_vals)
        model_phase_snow = resample_field(phase_model, "snow_mm_h", lat_vals, lon_vals)
        model_phase_mixed = resample_field(phase_model, "mixed_mm_h", lat_vals, lon_vals)

    for frame, weight in selected:
        if frame_is_model(frame):
            continue
        if not intersects_bounds(tile_bounds, frame.bounds):
            continue
        frame_rain = resample_field(frame, "rain_mm_h", lat_vals, lon_vals)
        frame_snow = resample_field(frame, "snow_mm_h", lat_vals, lon_vals)
        frame_mixed = resample_field(frame, "mixed_mm_h", lat_vals, lon_vals)
        if frame.metadata.get("phase_from_model") and model_phase_rain is not None:
            frame_rain, frame_snow, frame_mixed = split_precip_by_model_phase(
                frame_rain,
                model_phase_rain,
                model_phase_snow,
                model_phase_mixed,
            )
        rain = merge_max(rain, frame_rain, weight)
        snow = merge_max(snow, frame_snow, weight)
        mixed = merge_max(mixed, frame_mixed, weight)
        provider_ids.add(frame.provider_id)
        attributions.add(frame.attribution)
        source_urls.add(frame.source_url)
        if frame.valid_time:
            ref_times.append(frame.valid_time)

    model_active = False
    blend_weight = model_blend_weight(offset_minutes)
    if model_frames and tile_has_model_permission(tile_bounds, observation_bounds):
        exact = [frame for frame in model_frames if frame.lead_minutes == offset_minutes]
        if exact:
            model = exact[0]
            if intersects_bounds(tile_bounds, model.bounds):
                model_rain = resample_field(model, "rain_mm_h", lat_vals, lon_vals)
                model_snow = resample_field(model, "snow_mm_h", lat_vals, lon_vals)
                model_mixed = resample_field(model, "mixed_mm_h", lat_vals, lon_vals)
                model_weight = 1.0 if not np.any(rain + snow + mixed > 0) else blend_weight
                rain = merge_max(rain, model_rain, model_weight)
                snow = merge_max(snow, model_snow, model_weight)
                mixed = merge_max(mixed, model_mixed, model_weight)
                model_active = True
                provider_ids.add(model.provider_id)
                attributions.add(model.attribution)
                source_urls.add(model.source_url)
                if model.valid_time:
                    ref_times.append(model.valid_time)

    return WorkGrid(
        rain=rain,
        snow=snow,
        mixed=mixed,
        model_active=model_active,
        source_provider_ids=sorted(provider_ids),
        source_attributions=sorted(attributions),
        source_urls=sorted(source_urls),
        ref_times=ref_times,
    )


def product_values(grid: WorkGrid, product_id: str) -> np.ndarray:
    if product_id == RADAR_RAIN_RATE_ID:
        return grid.rain
    if product_id == RADAR_SNOW_RATE_ID:
        return grid.snow
    if product_id == RADAR_MIXED_RATE_ID:
        return grid.mixed
    if product_id == RADAR_PRECIP_RATE_ID:
        encoded = np.maximum.reduce([grid.rain, grid.snow, grid.mixed]).astype(np.float32)
        encoded = np.where(grid.snow > np.maximum(grid.rain, grid.mixed), -grid.snow, encoded)
        encoded = np.where(grid.mixed > np.maximum(grid.rain, grid.snow), grid.mixed + 1000.0, encoded)
        return encoded.astype(np.float32)
    raise ValueError(f"Unknown product: {product_id}")


def product_name(product_id: str) -> str:
    return {
        RADAR_RAIN_RATE_ID: "Radar Rain Rate",
        RADAR_SNOW_RATE_ID: "Radar Snow Rate",
        RADAR_MIXED_RATE_ID: "Radar Mixed Precipitation Rate",
        RADAR_PRECIP_RATE_ID: "Radar Total Precipitation Rate",
    }[product_id]


def product_precip_type(product_id: str) -> str:
    return {
        RADAR_RAIN_RATE_ID: "rain",
        RADAR_SNOW_RATE_ID: "snow",
        RADAR_MIXED_RATE_ID: "mixed",
        RADAR_PRECIP_RATE_ID: "all",
    }[product_id]


def product_palette(product_id: str) -> List[Dict[str, Any]]:
    if product_id == RADAR_SNOW_RATE_ID:
        return RAINBOW_SNOW_COLOR_STOPS
    if product_id == RADAR_MIXED_RATE_ID:
        return RAINBOW_MIXED_COLOR_STOPS
    return RAINBOW_RAIN_COLOR_STOPS


def max_pool(values: np.ndarray, step: int) -> np.ndarray:
    rows = math.ceil(values.shape[0] / step)
    cols = math.ceil(values.shape[1] / step)
    pooled = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            block = values[row * step : min((row + 1) * step, values.shape[0]), col * step : min((col + 1) * step, values.shape[1])]
            pooled[row, col] = float(np.nanmax(block)) if block.size else 0.0
    return pooled


def build_overview_tile(lon_vals: np.ndarray, lat_vals: np.ndarray, values: np.ndarray) -> Optional[bytes]:
    overview_lat = lat_vals[::OVERVIEW_SAMPLE_STEP]
    overview_lon = lon_vals[::OVERVIEW_SAMPLE_STEP]
    overview_values = max_pool(values, OVERVIEW_SAMPLE_STEP)[: len(overview_lat), : len(overview_lon)]
    if overview_values.size == 0 or overview_lat.size == 0 or overview_lon.size == 0:
        return None
    dy = (overview_lat[-1] - overview_lat[0]) / (len(overview_lat) - 1) if len(overview_lat) > 1 else -RADAR_TARGET_DEGREES
    dx = (overview_lon[-1] - overview_lon[0]) / (len(overview_lon) - 1) if len(overview_lon) > 1 else RADAR_TARGET_DEGREES
    return build_binary_tile_bytes(overview_lon, overview_lat, dx, dy, overview_values)


def generate_frame_products(
    frames: List[PrecipFrame],
    model_frames: List[PrecipFrame],
    offset_minutes: int,
) -> Dict[str, Dict[str, Any]]:
    selected = selected_frames_for_offset(frames, offset_minutes)
    observation_frames = [frame for frame in frames if not frame_is_model(frame)]
    observation_bounds = [frame.bounds for frame in observation_frames]
    tile_origins = set()
    for frame in observation_frames:
        for origin in bounds_tile_origins(frame.bounds):
            tile_origins.add(origin)
    if ENABLE_GLOBAL_MODEL_FALLBACK:
        for frame in model_frames:
            for origin in bounds_tile_origins(frame.bounds):
                tile_origins.add(origin)
    if not tile_origins:
        for frame in model_frames:
            for origin in bounds_tile_origins(frame.bounds):
                tile_origins.add(origin)

    product_pack_entries: Dict[str, Dict[Tuple[int, int], List[Tuple[str, bytes, int, int, np.ndarray, np.ndarray, np.ndarray]]]] = {
        product_id: {} for product_id in RADAR_SCALAR_LAYER_IDS
    }
    frame_stats = {
        product_id: {
            "tiles": 0,
            "packs": 0,
            "max_value": 0.0,
            "source_provider_ids": set(),
            "attributions": set(),
            "source_urls": set(),
            "ref_times": [],
            "model_active": False,
        }
        for product_id in RADAR_SCALAR_LAYER_IDS
    }

    for lat_start, lon_start in sorted(tile_origins):
        lat_end = min(lat_start + TILE_SIZE, 90)
        lon_end = min(lon_start + TILE_SIZE, 180)
        tile_bounds = (float(lon_start), float(lat_start), float(lon_end), float(lat_end))
        lat_vals = tile_axis(float(lat_start), float(lat_end), RADAR_TARGET_DEGREES, descending=True)
        lon_vals = tile_axis(float(lon_start), float(lon_end), RADAR_TARGET_DEGREES)
        if len(lat_vals) == 0 or len(lon_vals) == 0:
            continue
        grid = make_work_grid(selected, model_frames, offset_minutes, tile_bounds, lat_vals, lon_vals, observation_bounds)
        grid.rain = polish_precip(grid.rain)
        grid.snow = polish_precip(grid.snow)
        grid.mixed = polish_precip(grid.mixed)

        for product_id in RADAR_SCALAR_LAYER_IDS:
            values = product_values(grid, product_id)
            finite_abs = np.abs(np.where(np.isfinite(values), values, 0.0))
            if float(np.nanmax(finite_abs)) <= 0.0:
                continue
            dy = (lat_vals[-1] - lat_vals[0]) / (len(lat_vals) - 1) if len(lat_vals) > 1 else -RADAR_TARGET_DEGREES
            dx = (lon_vals[-1] - lon_vals[0]) / (len(lon_vals) - 1) if len(lon_vals) > 1 else RADAR_TARGET_DEGREES
            tile_bytes = build_binary_tile_bytes(lon_vals, lat_vals, dx, dy, values)
            pack_start = pack_origin(lat_start, lon_start)
            product_pack_entries[product_id].setdefault(pack_start, []).append(
                (tile_key(lat_start, lon_start), tile_bytes, lat_start, lon_start, lon_vals, lat_vals, values)
            )
            stats = frame_stats[product_id]
            stats["tiles"] += 1
            stats["max_value"] = max(float(stats["max_value"]), float(np.nanmax(finite_abs)))
            stats["source_provider_ids"].update(grid.source_provider_ids)
            stats["attributions"].update(grid.source_attributions)
            stats["source_urls"].update(grid.source_urls)
            stats["ref_times"].extend(grid.ref_times)
            stats["model_active"] = bool(stats["model_active"] or grid.model_active)

    frame_results: Dict[str, Dict[str, Any]] = {}
    for product_id in RADAR_SCALAR_LAYER_IDS:
        product_dir = os.path.join(OUTPUT_DIR, product_id)
        pack_count = 0
        for (pack_lat_start, pack_lon_start), pack_tiles in product_pack_entries[product_id].items():
            entries = [(key, tile_bytes) for key, tile_bytes, *_ in pack_tiles]
            all_lon = np.concatenate([item[4] for item in pack_tiles])
            all_lat = np.concatenate([item[5] for item in pack_tiles])
            lon_vals = np.unique(np.round(all_lon, 8))
            lat_vals = np.unique(np.round(all_lat, 8))[::-1]
            if len(lon_vals) > 0 and len(lat_vals) > 0:
                overview = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float32)
                for _, _, _, _, tile_lon, tile_lat, tile_values in pack_tiles:
                    lat_lookup = {round(float(v), 8): i for i, v in enumerate(lat_vals)}
                    lon_lookup = {round(float(v), 8): i for i, v in enumerate(lon_vals)}
                    row_indices = [lat_lookup[round(float(v), 8)] for v in tile_lat]
                    col_indices = [lon_lookup[round(float(v), 8)] for v in tile_lon]
                    overview[np.ix_(row_indices, col_indices)] = tile_values
                overview_tile = build_overview_tile(lon_vals, lat_vals, overview)
                if overview_tile:
                    entries.append((overview_tile_key(pack_lat_start, pack_lon_start), overview_tile))
            pack_path = os.path.join(product_dir, pack_filename_minutes(product_id, offset_minutes, pack_lat_start, pack_lon_start))
            write_tile_pack(pack_path, entries)
            pack_count += 1
        stats = frame_stats[product_id]
        stats["packs"] = pack_count
        ref_times = sorted({value for value in stats["ref_times"] if value})
        frame_results[product_id] = {
            "forecast_offset_minutes": offset_minutes,
            "status": "ok" if stats["tiles"] else "configuration_required",
            "tiles": int(stats["tiles"]),
            "packs": int(stats["packs"]),
            "max_value": round(float(stats["max_value"]), 3),
            "ref_time": ref_times[-1] if ref_times else None,
            "source_provider_ids": sorted(stats["source_provider_ids"]),
            "attribution": " / ".join(sorted(stats["attributions"])) if stats["attributions"] else None,
            "source_url": " / ".join(sorted(stats["source_urls"])) if stats["source_urls"] else RADAR_SOURCE_REFERENCE,
            "model_fallback_active": bool(stats["model_active"]),
        }
    return frame_results


def aggregate_product_results(frame_results_by_offset: Dict[int, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    products = []
    for product_id in RADAR_SCALAR_LAYER_IDS:
        frames = [frame_results_by_offset[offset][product_id] for offset in FORECAST_OFFSETS_MINUTES]
        total_tiles = sum(frame["tiles"] for frame in frames)
        total_packs = sum(frame["packs"] for frame in frames)
        max_value = max((frame["max_value"] for frame in frames), default=0.0)
        provider_ids = sorted({provider for frame in frames for provider in frame["source_provider_ids"]})
        attributions = sorted({frame["attribution"] for frame in frames if frame.get("attribution")})
        source_urls = sorted({frame["source_url"] for frame in frames if frame.get("source_url")})
        ref_times = sorted({frame["ref_time"] for frame in frames if frame.get("ref_time")})
        products.append(
            {
                "id": product_id,
                "name": product_name(product_id),
                "status": "ok" if total_tiles else "configuration_required",
                "region": "Global available radar coverage",
                "bounds": [-180.0, -90.0, 180.0, 90.0],
                "ref_time": ref_times[-1] if ref_times else None,
                "tiles": total_tiles,
                "packs": total_packs,
                "max_value": max_value,
                "units": RADAR_VALUE_UNITS,
                "value_parameter": RADAR_VALUE_PARAMETER,
                "value_units": RADAR_VALUE_UNITS,
                "path": product_id,
                "attribution": " / ".join(attributions) if attributions else None,
                "source_url": " / ".join(source_urls) if source_urls else RADAR_SOURCE_REFERENCE,
                "provider_url": " / ".join(source_urls) if source_urls else RADAR_SOURCE_REFERENCE,
                "forecast_offsets_minutes": FORECAST_OFFSETS_MINUTES,
                "frames": frames,
                "tile_format": TILE_FORMAT,
                "pack_format": PACK_FORMAT,
                "metadata": {
                    "parameter": RADAR_VALUE_PARAMETER,
                    "units": RADAR_VALUE_UNITS,
                    "value_range": [0, max(1.0, max_value)],
                    "target_degrees": RADAR_TARGET_DEGREES,
                    "client_coloring": True,
                    "transparent_zero": True,
                    "recommended_color_stops": product_palette(product_id),
                    "overview_reduction": "max",
                    "source_provider_ids": provider_ids,
                    "precip_type": product_precip_type(product_id),
                    "forecast_offsets_minutes": FORECAST_OFFSETS_MINUTES,
                    "nowcast_method": "provider_extrapolation_or_persistence_with_model_blend",
                    "model_blend_start_minutes": MODEL_BLEND_START_MINUTES,
                    "persistence_end_minutes": PERSISTENCE_END_MINUTES,
                    "quality_pipeline": {
                        "noise_floor_mm_h": RADAR_NOISE_FLOOR_MM_H,
                        "edge_threshold_mm_h": RADAR_EDGE_THRESHOLD_MM_H,
                        "speckle_filter": {
                            "max_value_mm_h": RADAR_SPECKLE_MAX_VALUE_MM_H,
                            "min_neighbors": RADAR_SPECKLE_MIN_NEIGHBORS,
                        },
                        "soft_dilation_passes": RADAR_DILATION_PASSES,
                        "smoothing_passes": RADAR_SMOOTHING_PASSES,
                    },
                    "wet_bulb_thresholds_c": {
                        "snow_lte": PRECIP_TYPE_SNOW_WET_BULB_C,
                        "mixed_gt": PRECIP_TYPE_SNOW_WET_BULB_C,
                        "mixed_lte": PRECIP_TYPE_RAIN_WET_BULB_C,
                        "rain_gt": PRECIP_TYPE_RAIN_WET_BULB_C,
                    },
                },
            }
        )
    return products


def provider_manifest(result: ProviderResult) -> Dict[str, Any]:
    frame_count = len(result.frames)
    bounds = None
    if result.frames:
        west = min(frame.bounds[0] for frame in result.frames)
        south = min(frame.bounds[1] for frame in result.frames)
        east = max(frame.bounds[2] for frame in result.frames)
        north = max(frame.bounds[3] for frame in result.frames)
        bounds = [west, south, east, north]
    payload = {
        "id": result.id,
        "name": result.name,
        "status": result.status,
        "source_url": result.source_url,
        "provider_url": result.source_url,
        "attribution": result.attribution,
        "frames": frame_count,
        "bounds": bounds,
        "error": result.error,
        "metadata": result.metadata,
    }
    return {key: value for key, value in payload.items() if value is not None}


def layer_entries(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "layers": [
            {
                "id": product["id"],
                "name": product["name"],
                "type": "gusty-grid-pack",
                "region": product["region"],
                "bounds": product["bounds"],
                "status": product["status"],
                "path": product["path"],
                "units": product["units"],
                "value_parameter": product["value_parameter"],
                "value_units": product["value_units"],
                "tiles": product["tiles"],
                "packs": product["packs"],
                "ref_time": product["ref_time"],
                "attribution": product["attribution"],
                "source_url": product["source_url"],
                "provider_url": product["provider_url"],
                "forecast_offsets_minutes": product["forecast_offsets_minutes"],
                "frames": product["frames"],
                "metadata": product["metadata"],
                "tile_format": product["tile_format"],
                "pack_format": product["pack_format"],
            }
            for product in products
        ],
    }


def vector_layer_entries() -> Dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "layers": [],
        "note": "The replacement radar pipeline publishes smoothed scalar grid packs; vector polygon radar packs are intentionally not generated.",
    }


def provider_catalog() -> Dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "providers": [
            {
                "id": "eccc_geomet",
                "name": "ECCC GeoMet North American radar rain/snow and extrapolation",
                "coverage": "North America",
                "free": True,
                "registration_required": False,
                "source_url": ECCC_RADAR_DOCS_URL,
            },
            {
                "id": "noaa_mrms",
                "name": "NOAA MRMS precipitation rate",
                "coverage": "United States / CONUS and adjacent regions",
                "free": True,
                "registration_required": False,
                "source_url": NOAA_MRMS_DOCS_URL,
                "metadata": {"phase_source": "model"},
            },
            {
                "id": "noaa_mrms_reflectivity",
                "name": "NOAA MRMS regional reflectivity",
                "coverage": "United States, Alaska, Hawaii, Caribbean, and Guam",
                "free": True,
                "registration_required": False,
                "source_url": NOAA_MRMS_DOCS_URL,
                "metadata": {
                    "native_kind": "GRIB2 radar reflectivity",
                    "native_parameter": "reflectivity_dbz",
                    "phase_source": "model",
                    "regions": sorted(MRMS_REFLECTIVITY_PRODUCTS),
                },
            },
            {
                "id": "iem_n0q",
                "name": "IEM NEXRAD regional radar composites",
                "coverage": "United States, Alaska, Hawaii, Caribbean, and Guam",
                "free": True,
                "registration_required": False,
                "source_url": IEM_BASE_URL,
                "metadata": {
                    "native_kind": "palette-indexed PNG radar reflectivity",
                    "native_parameter": "reflectivity_dbz",
                    "phase_source": "model",
                    "regions": sorted(IEM_REGIONS),
                },
            },
            {
                "id": "noaa_gfs",
                "name": "NOAA GFS short-range model fallback",
                "coverage": "Global model grid",
                "free": True,
                "registration_required": False,
                "source_url": "https://nomads.ncep.noaa.gov/",
                "metadata": {
                    "enabled_by_default": ENABLE_GLOBAL_MODEL_FALLBACK,
                    "forecast_hours_requested": NOAA_GFS_FORECAST_HOURS,
                    "display_offsets_minutes": FORECAST_OFFSETS_MINUTES,
                },
            },
            {
                "id": "dwd_open_data",
                "name": "DWD Open Data radar composites",
                "coverage": "Germany / Central Europe",
                "free": True,
                "registration_required": False,
                "source_url": DWD_SOURCE_URL,
            },
            {
                "id": "fmi_open_data",
                "name": "FMI Open Data Finland radar rain rate",
                "coverage": "Finland and nearby northern Europe",
                "free": True,
                "registration_required": False,
                "source_url": FMI_OPEN_DATA_URL,
                "license": "CC BY 4.0",
                "license_url": FMI_LICENSE_URL,
                "metadata": {
                    "wms_url": FMI_WMS_URL,
                    "wms_layer": FMI_RAIN_RATE_LAYER,
                    "native_kind": "WMS radar rain-rate image",
                    "phase_source": "model",
                },
            },
            {
                "id": "jma_nowcast",
                "name": "JMA nowcast radar tiles",
                "coverage": "Japan",
                "free": True,
                "registration_required": False,
                "source_url": "https://www.jma.go.jp/jp/radnowc/index.html",
                "metadata": {"native_kind": "XYZ radar precipitation tiles", "phase_source": "model"},
            },
            {
                "id": "marn_el_salvador",
                "name": "MARN/SNET El Salvador radar",
                "coverage": "El Salvador and nearby Central America",
                "free": True,
                "registration_required": False,
                "source_url": "https://storage.googleapis.com/radar-images-sv",
                "metadata": {"native_kind": "PNG radar reflectivity", "phase_source": "model"},
            },
            {
                "id": "cwa_taiwan",
                "name": "CWA Taiwan QPESUMS radar",
                "coverage": "Taiwan and surrounding waters",
                "free": True,
                "registration_required": False,
                "source_url": "https://opendata.cwa.gov.tw/",
                "metadata": {"native_kind": "XML radar reflectivity grid", "phase_source": "model"},
            },
            {
                "id": "met_malaysia",
                "name": "MET Malaysia radar composite",
                "coverage": "Peninsular Malaysia, Borneo, Brunei, Singapore, and N. Sumatra",
                "free": True,
                "registration_required": False,
                "source_url": "https://api.met.gov.my/",
                "metadata": {"native_kind": "animated GIF radar reflectivity", "phase_source": "model"},
            },
            {
                "id": "opera_europe",
                "name": "EUMETNET OPERA Europe radar",
                "coverage": "Pan-European radar composite",
                "free": True,
                "registration_required": False,
                "source_url": "https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/",
                "metadata": {"native_kind": "ODIM HDF5 radar reflectivity", "phase_source": "model"},
            },
            {
                "id": "dpc_italy",
                "name": "Radar-DPC Italy VMI radar",
                "coverage": "Italy and neighbouring areas",
                "free": True,
                "registration_required": False,
                "source_url": "https://radar-api.protezionecivile.it",
                "license": "CC-BY-SA 4.0",
                "metadata": {"native_kind": "GeoTIFF radar reflectivity", "phase_source": "model"},
            },
            {
                "id": "configured_providers",
                "name": "Configured free-key or licensed providers",
                "coverage": "Configured",
                "free": True,
                "registration_required": True,
                "source_url": "RADAR_PROVIDER_REGISTRY_FILE / RADAR_PROVIDER_REGISTRY_URL / RADAR_PROVIDERS_JSON",
                "metadata": {
                    "registry_file": RADAR_PROVIDER_REGISTRY_FILE,
                    "registry_url_configured": bool(RADAR_PROVIDER_REGISTRY_URL),
                    "allow_unverified_providers": RADAR_ALLOW_UNVERIFIED_PROVIDERS,
                },
                "supported_secret_names": [
                    "METOFFICE_DATAHUB_API_KEY",
                    "AEMET_API_KEY",
                    "METEOFRANCE_API_KEY",
                    "CWA_API_KEY",
                    "EUMETSAT_CONSUMER_KEY",
                    "EUMETSAT_CONSUMER_SECRET",
                ],
            },
        ],
    }


def run(print_summary: bool = False) -> Dict[str, Any]:
    ensure_clean_output()
    client = http_session()
    provider_results = fetch_all_providers(client)
    all_frames = [frame for result in provider_results for frame in result.frames]
    model_frames = [frame for frame in all_frames if frame_is_model(frame)]
    observation_frames = [frame for frame in all_frames if not frame_is_model(frame)]
    if not observation_frames and not model_frames:
        errors = [f"{result.id}: {result.error}" for result in provider_results if result.error]
        raise RuntimeError("No radar or model provider produced usable frames. " + " | ".join(errors))

    frame_results_by_offset: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for offset in FORECAST_OFFSETS_MINUTES:
        frame_results_by_offset[offset] = generate_frame_products(observation_frames, model_frames, offset)

    products = aggregate_product_results(frame_results_by_offset)
    active_provider_ids = sorted({frame.provider_id for frame in observation_frames})
    manifest = {
        "generated_at": now_iso(),
        "layers_file": "layers/radar_layers.json",
        "vector_layers_file": "layers/radar_vector_layers.json",
        "reference_basis": RADAR_SOURCE_REFERENCE,
        "value_parameter": RADAR_VALUE_PARAMETER,
        "value_units": RADAR_VALUE_UNITS,
        "forecast_offsets_minutes": FORECAST_OFFSETS_MINUTES,
        "active_observation_provider_count": len(active_provider_ids),
        "active_observation_provider_ids": active_provider_ids,
        "provider_count": len(provider_results),
        "provider_error_count": len([result for result in provider_results if result.status == "error"]),
        "recommended_color_stops": RAINBOW_RAIN_COLOR_STOPS,
        "rainbow_color_stops": {
            "rain": RAINBOW_RAIN_COLOR_STOPS,
            "snow": RAINBOW_SNOW_COLOR_STOPS,
            "mixed": RAINBOW_MIXED_COLOR_STOPS,
        },
        "pipeline_context": {
            "replacement": True,
            "transparent_zero": True,
            "forecast_window_minutes": [0, 120],
            "display_offsets_minutes": FORECAST_OFFSETS_MINUTES,
            "quality_goal": "smooth transparent rain/snow fields comparable to Rainbow.ai map overlays",
            "provider_failure_policy": "provider failures are warnings unless every provider fails",
            "allow_unverified_providers": RADAR_ALLOW_UNVERIFIED_PROVIDERS,
            "model_fallback_scope": (
                "global"
                if ENABLE_GLOBAL_MODEL_FALLBACK
                else "limited to observed radar coverage because RADAR_ENABLE_GLOBAL_MODEL_FALLBACK is false"
            ),
            "source_cadence_note": (
                "The app-facing layer exposes 15-minute offsets through 120 minutes. "
                "Radar providers may update more frequently or less frequently by region; global fallback is NOAA GFS "
                "hourly forecast data interpolated to those display offsets."
            ),
        },
        "products": products,
        "providers": [provider_manifest(result) for result in provider_results],
    }
    write_json(os.path.join(RADAR_DIR, "manifest.json"), manifest)
    write_json(os.path.join(RADAR_DIR, "provider_catalog.json"), provider_catalog())
    write_json(os.path.join(LAYER_DIR, "radar_layers.json"), layer_entries(products))
    write_json(os.path.join(LAYER_DIR, "radar_vector_layers.json"), vector_layer_entries())
    if print_summary:
        print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    run(print_summary=args.print_summary)


if __name__ == "__main__":
    main()
