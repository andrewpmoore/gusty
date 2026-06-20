import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import zipfile
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import requests


OUTPUT_DIR = "public/data"
HAZARD_DIR = os.path.join(OUTPUT_DIR, "hazards")
LAYER_DIR = os.path.join(OUTPUT_DIR, "layers")
TIMEOUT = 45
USER_AGENT = os.getenv(
    "GUSTY_USER_AGENT",
    "GustyWeather/1.0 (https://gustyweather.com; weather-data-ingestion)",
)

NOAA_ALERTS_URL = "https://api.weather.gov/alerts/active"
GDACS_RSS_URL = "https://www.gdacs.org/xml/rss.xml"
NHC_KML_URLS = [
    "https://www.nhc.noaa.gov/gis/kml/nhc.kmz",
    "https://www.nhc.noaa.gov/gis/kml/nhc_active.kml",
]
DEFAULT_METEOALARM_FEEDS = [
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-andorra",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-austria",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-belgium",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-bosnia-herzegovina",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-bulgaria",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-croatia",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-cyprus",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-czechia",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-denmark",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-estonia",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-finland",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-france",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-germany",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-greece",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-hungary",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-iceland",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-ireland",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-israel",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-italy",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-latvia",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-lithuania",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-luxembourg",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-malta",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-moldova",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-montenegro",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-netherlands",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-republic-of-north-macedonia",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-norway",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-poland",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-portugal",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-romania",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-serbia",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-slovakia",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-slovenia",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-spain",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-sweden",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-switzerland",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-ukraine",
    "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-united-kingdom",
]

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cap": "urn:oasis:names:tc:emergency:cap:1.2",
    "georss": "http://www.georss.org/georss",
    "kml": "http://www.opengis.net/kml/2.2",
}

SEVERITY_ORDER = {
    "unknown": 0,
    "minor": 1,
    "moderate": 2,
    "severe": 3,
    "extreme": 4,
}


class ProviderResult:
    def __init__(
        self,
        provider: str,
        features: Optional[List[Dict[str, Any]]] = None,
        status: str = "ok",
        error: Optional[str] = None,
        registration_required: bool = False,
        source_url: Optional[str] = None,
    ):
        self.provider = provider
        self.features = active_features(features or [])
        self.status = status
        self.error = error
        self.registration_required = registration_required
        self.source_url = source_url

    def manifest(self) -> Dict[str, Any]:
        data = {
            "status": self.status,
            "features": len(self.features),
            "registration_required": self.registration_required,
        }
        if self.error:
            data["error"] = self.error
        if self.source_url:
            data["source_url"] = self.source_url
        return data


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def session() -> requests.Session:
    client = requests.Session()
    client.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/geo+json, application/json, application/xml, text/xml, */*",
        }
    )
    return client


def get_text(client: requests.Session, url: str) -> str:
    response = client.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text


def get_bytes(client: requests.Session, url: str) -> bytes:
    response = client.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.content


def parse_iso(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()
    except ValueError:
        return text


def parse_datetime(value: Any) -> Optional[dt.datetime]:
    text = parse_iso(value)
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def active_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current = dt.datetime.now(dt.timezone.utc)
    active = []
    for feature in features:
        expires = feature.get("properties", {}).get("expires")
        expires_at = parse_datetime(expires)
        if expires_at and expires_at < current:
            continue
        active.append(feature)
    return active


def compact_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def severity(value: Any) -> str:
    text = compact_text(value)
    if not text:
        return "unknown"
    normalized = text.lower()
    if normalized in SEVERITY_ORDER:
        return normalized
    if normalized in {"yellow", "low"}:
        return "minor"
    if normalized in {"orange", "medium"}:
        return "moderate"
    if normalized in {"red", "high"}:
        return "severe"
    return "unknown"


def categorize_event(event: Any) -> str:
    text = (compact_text(event) or "").lower()
    if any(token in text for token in ["hurricane", "typhoon", "cyclone", "tropical storm"]):
        return "tropical"
    if any(token in text for token in ["fire", "wildfire", "forest fire"]):
        return "fire"
    if any(token in text for token in ["flood", "flash flood", "coastal flood"]):
        return "flood"
    if any(token in text for token in ["tornado", "thunderstorm", "convective", "hail"]):
        return "storm"
    if any(token in text for token in ["snow", "ice", "blizzard", "winter"]):
        return "winter"
    if any(token in text for token in ["heat", "hot", "cold", "freeze"]):
        return "temperature"
    if any(token in text for token in ["wind", "gale", "storm force"]):
        return "wind"
    return "weather"


def clean_props(props: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = compact_text(value)
            if not value:
                continue
        cleaned[key] = value
    return cleaned


def make_feature(
    provider: str,
    geometry: Optional[Dict[str, Any]],
    props: Dict[str, Any],
) -> Dict[str, Any]:
    event = props.get("event") or props.get("headline") or props.get("title")
    normalized = {
        "id": props.get("id"),
        "provider": provider,
        "sourceUrl": props.get("sourceUrl"),
        "category": props.get("category") or categorize_event(event),
        "event": event,
        "severity": severity(props.get("severity")),
        "urgency": compact_text(props.get("urgency")),
        "certainty": compact_text(props.get("certainty")),
        "status": compact_text(props.get("status")),
        "headline": compact_text(props.get("headline") or props.get("title")),
        "description": compact_text(props.get("description") or props.get("summary")),
        "instruction": compact_text(props.get("instruction")),
        "areaName": compact_text(props.get("areaName")),
        "country": compact_text(props.get("country")),
        "region": compact_text(props.get("region")),
        "effective": parse_iso(props.get("effective")),
        "onset": parse_iso(props.get("onset")),
        "expires": parse_iso(props.get("expires")),
        "updated": parse_iso(props.get("updated")),
        "layer": props.get("layer") or "alerts",
        "priority": SEVERITY_ORDER.get(severity(props.get("severity")), 0),
        "attribution": props.get("attribution"),
    }
    if not normalized["id"]:
        normalized["id"] = stable_id(provider, normalized)
    return {"type": "Feature", "geometry": geometry, "properties": clean_props(normalized)}


def stable_id(provider: str, props: Dict[str, Any]) -> str:
    parts = [
        provider,
        str(props.get("event") or ""),
        str(props.get("areaName") or ""),
        str(props.get("onset") or props.get("effective") or props.get("updated") or ""),
    ]
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", "|".join(parts)).strip("-")[:180]


def collection(features: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def read_previous_collection(previous_dir: Optional[str], relative_path: str) -> Dict[str, Any]:
    candidates = []
    if previous_dir:
        candidates.append(os.path.join(previous_dir, relative_path))
    candidates.append(os.path.join(relative_path))
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if data.get("type") == "FeatureCollection":
                    return data
            except (OSError, json.JSONDecodeError):
                continue
    return collection([])


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)


def merge_previous_failed(
    previous_dir: Optional[str],
    relative_path: str,
    fresh_features: List[Dict[str, Any]],
    failed_providers: Iterable[str],
) -> List[Dict[str, Any]]:
    failed = set(failed_providers)
    if not failed:
        return fresh_features
    previous = read_previous_collection(previous_dir, relative_path)
    preserved = [
        feature
        for feature in previous.get("features", [])
        if feature.get("properties", {}).get("provider") in failed
    ]
    for feature in preserved:
        feature.setdefault("properties", {})["stale"] = True
    return fresh_features + preserved


def parse_cap_polygons(parent: ET.Element) -> List[Dict[str, Any]]:
    polygons = []
    for element in parent.iter():
        if element.tag.endswith("polygon") and element.text:
            points = []
            for item in element.text.split():
                parts = item.split(",")
                if len(parts) < 2:
                    continue
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                except ValueError:
                    continue
                points.append([lon, lat])
            if len(points) >= 3:
                if points[0] != points[-1]:
                    points.append(points[0])
                polygons.append({"type": "Polygon", "coordinates": [points]})
    if len(polygons) == 1:
        return polygons
    if len(polygons) > 1:
        return [{"type": "MultiPolygon", "coordinates": [p["coordinates"] for p in polygons]}]
    return []


def parse_georss_geometry(parent: ET.Element) -> Optional[Dict[str, Any]]:
    point = parent.find(".//georss:point", NS)
    if point is not None and point.text:
        parts = point.text.split()
        if len(parts) >= 2:
            return {"type": "Point", "coordinates": [float(parts[1]), float(parts[0])]}
    polygon = parent.find(".//georss:polygon", NS)
    if polygon is not None and polygon.text:
        values = polygon.text.split()
        coords = []
        for index in range(0, len(values) - 1, 2):
            try:
                lat = float(values[index])
                lon = float(values[index + 1])
            except ValueError:
                continue
            coords.append([lon, lat])
        if len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return {"type": "Polygon", "coordinates": [coords]}
    return None


def parse_kml_coordinates(text: Optional[str]) -> List[List[float]]:
    coords = []
    if not text:
        return coords
    for item in text.split():
        parts = item.split(",")
        if len(parts) < 2:
            continue
        try:
            coords.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    return coords


def kml_placemark_geometry(placemark: ET.Element) -> Optional[Dict[str, Any]]:
    point = placemark.find(".//kml:Point/kml:coordinates", NS)
    if point is not None:
        coords = parse_kml_coordinates(point.text)
        if coords:
            return {"type": "Point", "coordinates": coords[0]}

    line = placemark.find(".//kml:LineString/kml:coordinates", NS)
    if line is not None:
        coords = parse_kml_coordinates(line.text)
        if len(coords) >= 2:
            return {"type": "LineString", "coordinates": coords}

    polygon = placemark.find(".//kml:Polygon//kml:LinearRing/kml:coordinates", NS)
    if polygon is not None:
        coords = parse_kml_coordinates(polygon.text)
        if len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return {"type": "Polygon", "coordinates": [coords]}

    return None


def element_text(parent: ET.Element, path: str) -> Optional[str]:
    item = parent.find(path, NS)
    return compact_text(item.text if item is not None else None)


def first_element_text(parent: ET.Element, paths: Iterable[str]) -> Optional[str]:
    for path in paths:
        text = element_text(parent, path)
        if text:
            return text
    return None


def fetch_noaa_alerts(client: requests.Session) -> ProviderResult:
    provider = "NOAA_NWS"
    try:
        response = client.get(NOAA_ALERTS_URL, timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        features = []
        for raw in payload.get("features", []):
            props = raw.get("properties", {})
            features.append(
                make_feature(
                    provider,
                    raw.get("geometry"),
                    {
                        "id": props.get("id") or raw.get("id"),
                        "sourceUrl": props.get("@id") or props.get("uri"),
                        "event": props.get("event"),
                        "severity": props.get("severity"),
                        "urgency": props.get("urgency"),
                        "certainty": props.get("certainty"),
                        "status": props.get("status"),
                        "headline": props.get("headline"),
                        "description": props.get("description"),
                        "instruction": props.get("instruction"),
                        "areaName": props.get("areaDesc"),
                        "country": "US",
                        "region": props.get("senderName"),
                        "effective": props.get("effective"),
                        "onset": props.get("onset"),
                        "expires": props.get("expires"),
                        "updated": props.get("sent"),
                        "attribution": "NOAA/National Weather Service",
                    },
                )
            )
        return ProviderResult(provider, features, source_url=NOAA_ALERTS_URL)
    except Exception as exc:
        return ProviderResult(provider, status="error", error=str(exc), source_url=NOAA_ALERTS_URL)


def meteoalarm_feeds() -> List[str]:
    configured = os.getenv("METEOALARM_FEEDS")
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return DEFAULT_METEOALARM_FEEDS


def fetch_meteoalarm(client: requests.Session) -> ProviderResult:
    provider = "METEOALARM"
    feeds = meteoalarm_feeds()
    features = []
    errors = []
    for feed_url in feeds:
        try:
            root = ET.fromstring(get_text(client, feed_url))
            entries = root.findall(".//atom:entry", NS)
            if not entries and root.tag.endswith("alert"):
                entries = [root]
            for entry in entries:
                polygons = parse_cap_polygons(entry)
                geometry = polygons[0] if polygons else parse_georss_geometry(entry)
                link = None
                for link_node in entry.findall("atom:link", NS):
                    href = link_node.attrib.get("href")
                    if href and (link is None or link_node.attrib.get("type") == "application/cap+xml"):
                        link = href
                features.append(
                    make_feature(
                        provider,
                        geometry,
                        {
                            "id": element_text(entry, "atom:id") or element_text(entry, "cap:identifier"),
                            "sourceUrl": link or feed_url,
                            "event": first_element_text(entry, ["cap:event", "cap:info/cap:event", "atom:title"]),
                            "severity": first_element_text(entry, ["cap:severity", "cap:info/cap:severity"]),
                            "urgency": first_element_text(entry, ["cap:urgency", "cap:info/cap:urgency"]),
                            "certainty": first_element_text(entry, ["cap:certainty", "cap:info/cap:certainty"]),
                            "status": element_text(entry, "cap:status"),
                            "headline": first_element_text(entry, ["cap:headline", "cap:info/cap:headline", "atom:title"]),
                            "description": first_element_text(entry, ["cap:description", "cap:info/cap:description", "atom:summary"]),
                            "instruction": first_element_text(entry, ["cap:instruction", "cap:info/cap:instruction"]),
                            "areaName": first_element_text(entry, ["cap:areaDesc", "cap:info/cap:area/cap:areaDesc"]),
                            "region": urlparse(feed_url).path.rsplit("-", 1)[-1],
                            "effective": first_element_text(entry, ["cap:effective", "cap:info/cap:effective"]),
                            "onset": first_element_text(entry, ["cap:onset", "cap:info/cap:onset"]),
                            "expires": first_element_text(entry, ["cap:expires", "cap:info/cap:expires"]),
                            "updated": element_text(entry, "atom:updated") or element_text(entry, "cap:sent"),
                            "attribution": "Meteoalarm/EUMETNET and participating national meteorological services",
                        },
                    )
                )
        except Exception as exc:
            errors.append(f"{feed_url}: {exc}")

    status = "ok" if not errors else ("partial" if features else "error")
    return ProviderResult(
        provider,
        features,
        status=status,
        error="; ".join(errors) if errors else None,
        source_url=",".join(feeds),
    )


def fetch_gdacs(client: requests.Session) -> ProviderResult:
    provider = "GDACS"
    try:
        root = ET.fromstring(get_text(client, GDACS_RSS_URL))
        features = []
        for item in root.findall(".//item"):
            title = element_text(item, "title")
            category = categorize_event(title)
            features.append(
                make_feature(
                    provider,
                    parse_georss_geometry(item),
                    {
                        "id": element_text(item, "guid") or element_text(item, "link"),
                        "sourceUrl": element_text(item, "link") or GDACS_RSS_URL,
                        "event": title,
                        "category": category,
                        "severity": element_text(item, "severity") or "unknown",
                        "headline": title,
                        "description": element_text(item, "description"),
                        "updated": element_text(item, "pubDate"),
                        "attribution": "Global Disaster Alert and Coordination System (GDACS)",
                    },
                )
            )
        return ProviderResult(provider, features, source_url=GDACS_RSS_URL)
    except Exception as exc:
        return ProviderResult(provider, status="error", error=str(exc), source_url=GDACS_RSS_URL)


def kml_document_from_bytes(payload: bytes, url: str) -> str:
    if url.endswith(".kmz") or payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            kml_names = [name for name in archive.namelist() if name.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("KMZ did not contain a KML document")
            return archive.read(kml_names[0]).decode("utf-8", errors="replace")
    return payload.decode("utf-8", errors="replace")


def fetch_nhc(client: requests.Session) -> ProviderResult:
    provider = "NOAA_NHC"
    features = []
    errors = []
    for url in NHC_KML_URLS:
        try:
            kml_text = kml_document_from_bytes(get_bytes(client, url), url)
            root = ET.fromstring(kml_text)
            for placemark in root.findall(".//kml:Placemark", NS):
                name = element_text(placemark, "kml:name")
                description = element_text(placemark, "kml:description")
                geometry = kml_placemark_geometry(placemark)
                if not name and not description and geometry is None:
                    continue
                features.append(
                    make_feature(
                        provider,
                        geometry,
                        {
                            "id": f"{url}#{name or len(features)}",
                            "sourceUrl": url,
                            "event": name or "Tropical cyclone product",
                            "category": "tropical",
                            "severity": "unknown",
                            "headline": name,
                            "description": description,
                            "layer": "tropical",
                            "attribution": "NOAA/National Hurricane Center",
                        },
                    )
                )
            if features:
                break
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    status = "ok" if features else ("error" if errors else "ok")
    return ProviderResult(
        provider,
        features,
        status=status,
        error="; ".join(errors) if errors and not features else None,
        source_url=",".join(NHC_KML_URLS),
    )


def fire_time(row: Dict[str, str]) -> Optional[str]:
    date = row.get("acq_date")
    raw_time = (row.get("acq_time") or "").zfill(4)
    if not date or len(raw_time) != 4:
        return None
    try:
        parsed = dt.datetime.strptime(f"{date} {raw_time}", "%Y-%m-%d %H%M")
        return parsed.replace(tzinfo=dt.timezone.utc).isoformat()
    except ValueError:
        return None


def fetch_firms(client: requests.Session) -> ProviderResult:
    provider = "NASA_FIRMS"
    map_key = os.getenv("FIRMS_MAP_KEY")
    if not map_key:
        return ProviderResult(
            provider,
            status="skipped",
            error="FIRMS_MAP_KEY is not set",
            registration_required=True,
            source_url="https://www.earthdata.nasa.gov/data/tools/firms",
        )

    products = [item.strip() for item in os.getenv("FIRMS_PRODUCTS", "VIIRS_SNPP_NRT").split(",") if item.strip()]
    days = os.getenv("FIRMS_DAYS", "1")
    features = []
    errors = []
    for product in products:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{product}/world/{days}"
        try:
            rows = csv.DictReader(io.StringIO(get_text(client, url)))
            for row in rows:
                try:
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                except (KeyError, ValueError):
                    continue
                confidence = row.get("confidence")
                features.append(
                    make_feature(
                        provider,
                        {"type": "Point", "coordinates": [lon, lat]},
                        {
                            "id": f"{product}:{row.get('latitude')}:{row.get('longitude')}:{row.get('acq_date')}:{row.get('acq_time')}",
                            "sourceUrl": "https://www.earthdata.nasa.gov/data/tools/firms",
                            "event": "Active fire detection",
                            "category": "fire",
                            "severity": "moderate" if str(confidence).lower() in {"h", "high"} else "unknown",
                            "headline": f"Active fire detection ({product})",
                            "description": f"Confidence: {confidence}; brightness: {row.get('bright_ti4') or row.get('brightness')}",
                            "updated": fire_time(row),
                            "layer": "fires",
                            "attribution": "NASA FIRMS",
                        },
                    )
                )
        except Exception as exc:
            errors.append(f"{product}: {exc}")

    status = "ok" if not errors else ("partial" if features else "error")
    return ProviderResult(
        provider,
        features,
        status=status,
        error="; ".join(errors) if errors else None,
        registration_required=True,
        source_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv",
    )


def fetch_optional_geojson_provider(
    client: requests.Session,
    provider: str,
    env_var: str,
    attribution: str,
) -> ProviderResult:
    url = os.getenv(env_var)
    if not url:
        return ProviderResult(provider, status="skipped", error=f"{env_var} is not set")
    try:
        payload = client.get(url, timeout=TIMEOUT)
        payload.raise_for_status()
        data = payload.json()
        features = []
        for raw in data.get("features", []):
            props = raw.get("properties", {})
            features.append(
                make_feature(
                    provider,
                    raw.get("geometry"),
                    {
                        "id": props.get("id") or raw.get("id"),
                        "sourceUrl": props.get("sourceUrl") or url,
                        "event": props.get("event") or props.get("headline") or props.get("title"),
                        "severity": props.get("severity"),
                        "urgency": props.get("urgency"),
                        "certainty": props.get("certainty"),
                        "status": props.get("status"),
                        "headline": props.get("headline") or props.get("title"),
                        "description": props.get("description"),
                        "instruction": props.get("instruction"),
                        "areaName": props.get("areaName") or props.get("areaDesc"),
                        "country": props.get("country"),
                        "region": props.get("region"),
                        "effective": props.get("effective"),
                        "onset": props.get("onset"),
                        "expires": props.get("expires"),
                        "updated": props.get("updated"),
                        "attribution": attribution,
                    },
                )
            )
        return ProviderResult(provider, features, source_url=url)
    except Exception as exc:
        return ProviderResult(provider, status="error", error=str(exc), source_url=url)


def raster_layers() -> Dict[str, Any]:
    generated_at = now_iso()
    return {
        "generated_at": generated_at,
        "layers": [
            {
                "id": "noaa_nws_alerts",
                "name": "NOAA/NWS Active Alerts",
                "type": "geojson",
                "coverage": "United States and territories",
                "url": NOAA_ALERTS_URL,
                "free": True,
                "registration_required": False,
                "notes": "Use normalized hazards/alerts.geojson in the app where possible.",
            },
            {
                "id": "noaa_nhc_gis",
                "name": "NOAA/NHC Tropical Cyclone GIS",
                "type": "kml/kmz/shapefile",
                "coverage": "Atlantic, Eastern Pacific, Central Pacific",
                "url": "https://www.nhc.noaa.gov/gis/",
                "free": True,
                "registration_required": False,
            },
            {
                "id": "eumetsat_eumetview_wms",
                "name": "EUMETSAT EUMETView WMS",
                "type": "wms",
                "coverage": "Europe, Africa, Atlantic and adjacent satellite sectors depending on product",
                "url": "https://view.eumetsat.int/geoserver/wms",
                "free": True,
                "registration_required": True,
                "notes": "Use only products/licences approved for the app; near-real-time products can require account/licence acceptance.",
            },
            {
                "id": "canada_msc_geomet_wms",
                "name": "Environment Canada MSC GeoMet WMS",
                "type": "wms",
                "coverage": "Canada",
                "url": "https://geo.weather.gc.ca/geomet",
                "free": True,
                "registration_required": False,
            },
            {
                "id": "nasa_firms",
                "name": "NASA FIRMS Active Fires",
                "type": "csv/api",
                "coverage": "Global",
                "url": "https://www.earthdata.nasa.gov/data/tools/firms",
                "free": True,
                "registration_required": True,
                "notes": "Set FIRMS_MAP_KEY to include this layer in hazards/fires.geojson.",
            },
            {
                "id": "gdacs",
                "name": "GDACS Global Disaster Alerts",
                "type": "rss/xml",
                "coverage": "Global",
                "url": GDACS_RSS_URL,
                "free": True,
                "registration_required": False,
            },
        ],
    }


def partition_outputs(results: List[ProviderResult]) -> Dict[str, List[Dict[str, Any]]]:
    alerts = []
    tropical = []
    fires = []
    floods = []

    for result in results:
        for feature in result.features:
            category = feature.get("properties", {}).get("category")
            if category == "tropical":
                tropical.append(feature)
            elif category == "fire":
                fires.append(feature)
            elif category == "flood":
                floods.append(feature)
            else:
                alerts.append(feature)

    return {
        "alerts.geojson": alerts,
        "tropical.geojson": tropical,
        "fires.geojson": fires,
        "floods.geojson": floods,
    }


def build_manifest(results: List[ProviderResult], outputs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "format": "gusty-hazards-v1",
        "refresh_target_minutes": 15,
        "user_agent": USER_AGENT,
        "outputs": {
            name: {"features": len(features), "path": f"hazards/{name}"}
            for name, features in outputs.items()
        },
        "providers": {result.provider: result.manifest() for result in results},
        "schema": {
            "geometry": "GeoJSON Feature geometry; may be null when source provides event-level data only",
            "properties": [
                "id",
                "provider",
                "sourceUrl",
                "category",
                "event",
                "severity",
                "urgency",
                "certainty",
                "status",
                "headline",
                "description",
                "instruction",
                "areaName",
                "country",
                "region",
                "effective",
                "onset",
                "expires",
                "updated",
                "layer",
                "priority",
                "attribution",
                "stale",
            ],
        },
    }


def run(previous_dir: Optional[str]) -> Dict[str, Any]:
    client = session()
    os.makedirs(HAZARD_DIR, exist_ok=True)
    os.makedirs(LAYER_DIR, exist_ok=True)

    results = [
        fetch_noaa_alerts(client),
        fetch_meteoalarm(client),
        fetch_gdacs(client),
        fetch_nhc(client),
        fetch_firms(client),
        fetch_optional_geojson_provider(
            client,
            "CANADA_MSC_ALERTS",
            "CANADA_MSC_ALERTS_GEOJSON_URL",
            "Environment and Climate Change Canada / MSC GeoMet",
        ),
    ]

    outputs = partition_outputs(results)
    failed_providers = [
        result.provider
        for result in results
        if result.status in {"error", "skipped"} and result.provider != "NASA_FIRMS"
    ]

    for name, features in outputs.items():
        relative = os.path.join("hazards", name)
        merged = merge_previous_failed(previous_dir, relative, features, failed_providers)
        outputs[name] = merged
        write_json(os.path.join(HAZARD_DIR, name), collection(merged))

    manifest = build_manifest(results, outputs)
    write_json(os.path.join(HAZARD_DIR, "manifest.json"), manifest)
    write_json(os.path.join(LAYER_DIR, "raster_layers.json"), raster_layers())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-dir", default=os.getenv("GUSTY_PREVIOUS_DATA_DIR"))
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    manifest = run(args.previous_dir)
    if args.print_summary:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
