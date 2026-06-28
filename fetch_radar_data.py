import argparse
import datetime as dt
import gzip
import io
import json
import math
import os
import re
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import requests
from PIL import Image


OUTPUT_DIR = "public/data"
RADAR_DIR = os.path.join(OUTPUT_DIR, "radar")
RADAR_VECTOR_DIR = os.path.join(OUTPUT_DIR, "radar_vectors")
LAYER_DIR = os.path.join(OUTPUT_DIR, "layers")
TIMEOUT = 90
USER_AGENT = os.getenv(
    "GUSTY_USER_AGENT",
    "GustyWeather/1.0 (https://gustyweather.com; weather-radar-ingestion)",
)

IEM_CURRENT_URL = "https://mesonet.agron.iastate.edu/data/gis/images/4326/USCOMP"
IEM_N0Q_DOCS_URL = "https://mesonet.agron.iastate.edu/docs/nexrad_composites/"
MRMS_BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
MRMS_PRODUCT_PREFIX = "CONUS/PrecipRate_00.00"
MRMS_DOCS_URL = "https://registry.opendata.aws/noaa-mrms/"
MET_OFFICE_PUBLIC_RADAR_URL = "https://www.metoffice.gov.uk/public/weather/observation/map/#?map=Rainfall"
MET_OFFICE_DATAHUB_URL = "https://datahub.metoffice.gov.uk/"
ECCC_GEOMET_URL = "https://geo.weather.gc.ca/geomet"
ECCC_RADAR_LAYER = "RADAR_1KM_RRAI"
ECCC_RADAR_STYLE = "RADARURPPRECIPR14-LINEAR"
JMA_NOWCAST_TIMES_URL = "https://www.jma.go.jp/bosai/jmatile/data/nowc/targetTimes_N1.json"
JMA_NOWCAST_TILE_TEMPLATE = "https://www.jma.go.jp/bosai/jmatile/data/nowc/{base_time}/none/{valid_time}/surf/hrpns/{z}/{x}/{y}.png"
OPERA_CIRRUS_LIST_IMAGES_URL = "https://cdn.fmi.fi/demos/eumetnet-web-site-radar-animator/list-images/"
OPERA_CIRRUS_ANIMATOR_URL = "https://www.eumetnet.eu/observations/opera-radar-animation/"
OPERA_CIRRUS_PRODUCT_URL = "https://www.eumetnet.eu/wp-content/uploads/2024/06/OPERA_Max-Reflectivity_Product-Sheet_Ed-2.0.pdf"
OPERA_NIMBUS_PRODUCT_URL = "https://www.eumetnet.eu/wp-content/uploads/2024/06/NIMBUS_datasheet_composites_1.0_13062024.pdf"
OPERA_NIMBUS_RAIN_RATE_PREFIX = "T_PAAH22_C_EUOC_"
OPERA_CIRRUS_REFLECTIVITY_PREFIX = "T_PABV21_C_EUOC_"

CONFIGURABLE_RADAR_REGIONS = [
    {
        "id": "radar_uk_configured",
        "name": "UK Radar Reflectivity",
        "region": "United Kingdom",
        "attribution": "Met Office",
        "source_url": MET_OFFICE_PUBLIC_RADAR_URL,
        "notes": "Met Office public maps are rendered imagery; use an approved DataHub or image feed with a declared palette mapping.",
    },
    {
        "id": "radar_ireland_configured",
        "name": "Ireland Radar Reflectivity",
        "region": "Ireland",
        "attribution": "Met Eireann / Met Office where applicable",
        "source_url": "https://www.met.ie/",
        "notes": "Configure an approved Met Eireann or Met Office image/WMS endpoint with a declared palette mapping.",
    },
    {
        "id": "radar_europe_opera_cirrus",
        "name": "Europe OPERA/CIRRUS Rain Rate",
        "region": "Europe",
        "attribution": "EUMETNET OPERA / Finnish Meteorological Institute",
        "source_url": OPERA_CIRRUS_ANIMATOR_URL,
        "notes": "Active public Europe layer. Replace with OPERA NIMBUS rain-rate raw products when a reachable ORD/EWC endpoint is available.",
    },
    {
        "id": "radar_canada_configured",
        "name": "Canada Radar Reflectivity",
        "region": "Canada",
        "attribution": "Environment and Climate Change Canada / MSC GeoMet",
        "source_url": "https://geo.weather.gc.ca/geomet",
        "notes": "MSC GeoMet is free and anonymous, but a layer name and legend-to-intensity mapping must be selected.",
    },
    {
        "id": "radar_australia_configured",
        "name": "Australia Radar Reflectivity",
        "region": "Australia",
        "attribution": "Bureau of Meteorology",
        "source_url": "http://www.bom.gov.au/australia/radar/",
        "notes": "Use an approved BoM radar image endpoint and palette mapping.",
    },
    {
        "id": "radar_japan_configured",
        "name": "Japan Radar Intensity",
        "region": "Japan",
        "attribution": "Japan Meteorological Agency",
        "source_url": "https://www.jma.go.jp/jp/radnowc/index.html",
        "notes": "Configure the selected JMA image endpoint and legend mapping.",
    },
    {
        "id": "radar_germany_configured",
        "name": "Germany Radar Intensity",
        "region": "Germany",
        "attribution": "Deutscher Wetterdienst",
        "source_url": "https://opendata.dwd.de/",
        "notes": "DWD open data can provide stronger numeric options; configure either a decoded image or preprocessed endpoint.",
    },
    {
        "id": "radar_netherlands_configured",
        "name": "Netherlands Radar Reflectivity",
        "region": "Netherlands",
        "attribution": "KNMI",
        "source_url": "https://data.knmi.nl/datasets/radar_reflectivity_composites/2.0",
        "notes": "KNMI radar reflectivity composites are a good numeric candidate; configure an accessible product endpoint.",
    },
    {
        "id": "radar_france_configured",
        "name": "France Radar Intensity",
        "region": "France",
        "attribution": "Meteo-France",
        "source_url": "https://donneespubliques.meteofrance.fr/",
        "notes": "Configure an approved Meteo-France public data product or image endpoint.",
    },
    {
        "id": "radar_spain_configured",
        "name": "Spain Radar Intensity",
        "region": "Spain",
        "attribution": "AEMET",
        "source_url": "https://opendata.aemet.es/",
        "notes": "AEMET OpenData often needs an API key; configure the endpoint and palette mapping if enabled.",
    },
    {
        "id": "radar_hong_kong_configured",
        "name": "Hong Kong Radar Intensity",
        "region": "Hong Kong",
        "attribution": "Hong Kong Observatory",
        "source_url": "http://maps.weather.gov.hk/gis-portal/web/index_e.html",
        "notes": "Configure the HKO radar endpoint and legend mapping.",
    },
    {
        "id": "radar_south_korea_configured",
        "name": "South Korea Radar Intensity",
        "region": "South Korea",
        "attribution": "Korea Meteorological Administration",
        "source_url": "http://www.kma.go.kr/weather/images/rader_integrate.jsp",
        "notes": "Configure the KMA radar endpoint and legend mapping.",
    },
    {
        "id": "radar_india_configured",
        "name": "India Radar Intensity",
        "region": "India",
        "attribution": "India Meteorological Department",
        "source_url": "https://mausam.imd.gov.in/imd_latest/contents/index_radar.php",
        "notes": "Configure selected IMD mosaic or regional radar endpoints.",
    },
    {
        "id": "radar_new_zealand_configured",
        "name": "New Zealand Radar Intensity",
        "region": "New Zealand",
        "attribution": "MetService New Zealand",
        "source_url": "https://about.metservice.com/our-company/about-this-site/open-access-data/",
        "notes": "Configure an approved MetService open access endpoint.",
    },
    {
        "id": "radar_brazil_configured",
        "name": "Brazil Radar Intensity",
        "region": "Brazil",
        "attribution": "Brazilian radar providers",
        "source_url": "https://www.redemet.aer.mil.br/",
        "notes": "Multiple regional radar providers are available; configure one composite or regional endpoint at a time.",
    },
    {
        "id": "radar_argentina_configured",
        "name": "Argentina Radar Intensity",
        "region": "Argentina",
        "attribution": "Servicio Meteorologico Nacional",
        "source_url": "https://www.smn.gob.ar/radar",
        "notes": "Configure the SMN radar endpoint and legend mapping.",
    },
]

ADDITIONAL_CONFIGURABLE_REGION_SOURCES = [
    ("Argentina", "Servicio Meteorologico Nacional", "https://www.smn.gob.ar/radar"),
    ("Australia", "Bureau of Meteorology", "http://www.bom.gov.au/australia/radar/"),
    ("Austria", "Austro Control", "https://www.austrocontrol.at/en/weather/weather_for_all/weather_radar"),
    ("Bahamas", "Bahamas Department of Meteorology", "https://radars.bahamasweather.org.bs"),
    ("Bahrain", "Bahrain Meteorological Directorate", "http://www.bahrainweather.gov.bh/web/guest/radar/"),
    ("Bangladesh", "Bangladesh Meteorological Department", "https://live8.bmd.gov.bd/"),
    ("Belgium", "Royal Meteorological Institute of Belgium", "http://www.meteo.be/meteo/view/en/123361-Radar.html"),
    ("Belize", "National Meteorological Service of Belize", "http://nms.gov.bz/sensors/radar-imagery/"),
    ("Bermuda", "Bermuda Weather Service", "http://www.weather.bm/tools/graphics.asp?name=250KM%20SRI"),
    ("Brazil", "Brazilian radar providers", "https://www.redemet.aer.mil.br/"),
    ("Bulgaria", "Bulgarian Hail Suppression Agency", "https://www.weathermod-bg.eu"),
    ("Cambodia", "Cambodia Ministry of Water Resources and Meteorology", "http://www.cambodiameteo.com/slideshow?menu=117&lang=km&domain=CAMBODIA"),
    ("Canada", "Environment and Climate Change Canada / MSC GeoMet", "https://geo.weather.gc.ca/geomet"),
    ("Cayman Islands", "Cayman Islands National Weather Service", "http://www.weather.gov.ky/portal/page/portal/nwshome/forecasthome/radar"),
    ("China", "China National Meteorological Center", "http://www.nmc.cn/publish/radar/chinaall.html"),
    ("Colombia", "Aeronautica Civil de Colombia", "http://meteorologia.aerocivil.gov.co/radar/"),
    ("Costa Rica", "Instituto Meteorologico Nacional de Costa Rica", "https://www.imn.ac.cr/radar"),
    ("Croatia", "Croatian Meteorological and Hydrological Service", "https://meteo.hr/podaci_e.php?section=podaci_mjerenja&param=radari&el=kompozit"),
    ("Cuba", "Instituto de Meteorologia de Cuba", "http://www.insmet.cu/asp/genesis.asp?TB0=PLANTILLAS&TB1=RADARES"),
    ("Curacao", "Meteorological Department Curacao", "http://www.meteo.cw/rad_still_ppi.php"),
    ("Cyprus", "Cyprus Department of Meteorology", "https://www.dom.org.cy/"),
    ("Czechia", "Czech Hydrometeorological Institute", "https://www.chmi.cz"),
    ("Denmark", "Danish Meteorological Institute", "https://www.dmi.dk/"),
    ("El Salvador", "Ministerio de Medio Ambiente y Recursos Naturales El Salvador", "http://www.snet.gob.sv/googlemaps/radares/radaresSV8.php"),
    ("Estonia", "Estonian Environment Agency", "https://www.ilmateenistus.ee/ilm/ilmavaatlused/radar/?#layers/precipitation"),
    ("Fiji", "Fiji Meteorological Service", "http://www.met.gov.fj/radar.php"),
    ("Finland", "Finnish Meteorological Institute", "http://en.ilmatieteenlaitos.fi/rain-and-cloudiness/finland"),
    ("France", "Meteo-France", "https://donneespubliques.meteofrance.fr/"),
    ("Germany", "Deutscher Wetterdienst", "https://opendata.dwd.de/"),
    ("Greece", "Hellenic National Meteorological Service", "http://www.hnms.gr/emy/el/observation/eikones_radar"),
    ("Guatemala", "INSIVUMEH", "http://www.insivumeh.gob.gt/?page_id=1048"),
    ("Guyana", "Hydrometeorological Service of Guyana", "http://hydromet.gov.gy/400km-radar-image/"),
    ("Hong Kong", "Hong Kong Observatory", "http://maps.weather.gov.hk/gis-portal/web/index_e.html"),
    ("Hungary", "Hungarian Meteorological Service", "http://www.met.hu/idojaras/aktualis_idojaras/radar/"),
    ("Iceland", "Icelandic Meteorological Office", "http://en.vedur.is/weather/observations/radar/"),
    ("India", "India Meteorological Department", "https://mausam.imd.gov.in/imd_latest/contents/index_radar.php"),
    ("Indonesia", "BMKG", "http://www.bmkg.go.id/cuaca/citra-radar.bmkg?lang=EN"),
    ("Ireland", "Met Eireann / Met Office where applicable", "https://www.met.ie/"),
    ("Israel", "Israel Meteorological Service", "https://ims.gov.il"),
    ("Italy", "Italian Civil Protection / regional radar providers", "https://mappe.protezionecivile.gov.it/it/mappe-rischi/piattaforma-radar"),
    ("Japan", "Japan Meteorological Agency", "https://www.jma.go.jp/jp/radnowc/index.html"),
    ("Kuwait", "Kuwait Meteorological Department", "http://www.met.gov.kw/Radar/maxz_250.php?lang=eng"),
    ("Latvia", "Latvian Environment, Geology and Meteorology Centre", "http://videscentrs.lvgmc.lv"),
    ("Lithuania", "Lithuanian Hydrometeorological Service", "http://www.meteo.lt/dabar/radarai/"),
    ("Malaysia", "Malaysian Meteorological Department", "https://www.met.gov.my/pencerapan/radar-malaysia/"),
    ("Malta", "Malta International Airport Weather", "https://www.maltairport.com/weather/radar-images/"),
    ("Mauritius", "Mauritius Meteorological Services", "http://radar.metservice.intnet.mu/radar-image-viewer/"),
    ("Mexico", "CONAGUA Servicio Meteorologico Nacional", "https://smn.conagua.gob.mx/"),
    ("Mongolia", "Mongolian weather service", "http://tsag-agaar.gov.mn/"),
    ("Myanmar", "Department of Meteorology and Hydrology Myanmar", "https://www.moezala.gov.mm/"),
    ("Netherlands", "KNMI", "https://data.knmi.nl/datasets/radar_reflectivity_composites/2.0"),
    ("New Zealand", "MetService New Zealand", "https://about.metservice.com/our-company/about-this-site/open-access-data/"),
    ("Oman", "Oman Directorate General of Meteorology", "http://www.met.gov.om/opencms/export/sites/default/dgman/en/weather-chart/map-data/index.html"),
    ("Pakistan", "Pakistan Meteorological Department", "http://www.pmd.gov.pk/Electronic-Met/Radar-images.html"),
    ("Panama", "Panama Canal Authority / IMHPA", "https://www.imhpa.gob.pa/es/radar-meteorologico"),
    ("Paraguay", "Direcccion de Meteorologia e Hidrologia Paraguay", "http://www.meteorologia.gov.py/radar/"),
    ("Philippines", "DOST-PAGASA", "https://www.panahon.gov.ph/"),
    ("Poland", "IMGW", "https://danepubliczne.imgw.pl/datastore"),
    ("Portugal", "IPMA", "http://www.ipma.pt/pt/otempo/obs.remote/index.jsp"),
    ("Romania", "Romanian National Meteorological Administration", "http://meteoromania.ro/anm2/radarm/radar.index.php"),
    ("Saudi Arabia", "Saudi radar provider", "https://www.radar-flask.xyz"),
    ("Serbia", "Republic Hydrometeorological Service of Serbia", "http://www.hidmet.gov.rs/ciril/osmotreni/radarska3.php"),
    ("Singapore", "Meteorological Service Singapore", "http://www.weather.gov.sg/weather-rain-area-50km/"),
    ("Slovakia", "Slovak Hydrometeorological Institute", "http://www.shmu.sk/en/?page=65&id="),
    ("Slovenia", "Slovenian Environment Agency", "http://www.arso.gov.si/vreme/napovedi%20in%20podatki/radar.html"),
    ("South Korea", "Korea Meteorological Administration", "http://www.kma.go.kr/weather/images/rader_integrate.jsp"),
    ("Spain", "AEMET", "https://opendata.aemet.es/"),
    ("Sweden", "SMHI", "https://smhi.se"),
    ("Switzerland", "Metradar", "http://www.metradar.ch/2009/pc/index15.php"),
    ("Taiwan", "Central Weather Administration Taiwan", "http://opendata.cwa.gov.tw/"),
    ("Thailand", "Thai Meteorological Department", "http://weather.tmd.go.th"),
    ("Trinidad and Tobago", "Trinidad and Tobago Meteorological Service", "http://www.metoffice.gov.tt/Radar_Imagery"),
    ("Turkey", "Turkish State Meteorological Service", "http://www.mgm.gov.tr/sondurum/radar.aspx?rG=img&rR=00&rU=ppi#sfB"),
    ("United Kingdom", "Met Office", MET_OFFICE_PUBLIC_RADAR_URL),
    ("Venezuela", "INAMEH", "http://www.inameh.gob.ve/web/"),
    ("Vietnam", "Vietnam meteorological radar service", "http://hymetnet.gov.vn/radar/"),
]


REGION_BOUNDS = {
    "Argentina": [-73.6, -55.1, -53.6, -21.8],
    "Australia": [112.9, -44.0, 154.0, -10.0],
    "Austria": [9.5, 46.3, 17.2, 49.1],
    "Bahamas": [-80.6, 20.8, -72.5, 27.3],
    "Bahrain": [50.3, 25.5, 50.9, 26.4],
    "Bangladesh": [88.0, 20.5, 92.8, 26.8],
    "Belgium": [2.5, 49.4, 6.4, 51.6],
    "Belize": [-89.3, 15.8, -87.7, 18.5],
    "Bermuda": [-65.2, 31.9, -64.4, 32.6],
    "Brazil": [-74.0, -34.0, -34.0, 5.4],
    "Bulgaria": [22.3, 41.2, 28.7, 44.3],
    "Cambodia": [102.3, 10.2, 107.7, 14.8],
    "Canada": [-141.0, 41.5, -52.5, 83.5],
    "Cayman Islands": [-81.7, 19.0, -79.7, 19.9],
    "China": [73.4, 18.1, 135.1, 53.6],
    "Colombia": [-79.0, -4.3, -66.8, 13.5],
    "Costa Rica": [-86.0, 8.0, -82.5, 11.3],
    "Croatia": [13.4, 42.1, 19.5, 46.6],
    "Cuba": [-85.1, 19.8, -74.1, 23.4],
    "Curacao": [-69.3, 11.8, -68.6, 12.5],
    "Cyprus": [32.0, 34.4, 34.8, 35.8],
    "Czechia": [12.0, 48.5, 18.9, 51.1],
    "Denmark": [7.8, 54.4, 15.3, 57.9],
    "El Salvador": [-90.2, 13.0, -87.7, 14.5],
    "Estonia": [21.5, 57.4, 28.3, 59.8],
    "Europe": [-25.0, 34.0, 45.0, 72.0],
    "Fiji": [176.5, -20.8, 180.0, -12.0],
    "Finland": [19.0, 59.5, 31.8, 70.2],
    "France": [-5.3, 41.1, 9.8, 51.3],
    "Germany": [5.5, 47.0, 15.5, 55.2],
    "Greece": [19.3, 34.6, 29.8, 41.8],
    "Guatemala": [-92.4, 13.6, -88.0, 17.9],
    "Guyana": [-61.5, 1.1, -56.4, 8.6],
    "Hong Kong": [113.8, 22.1, 114.5, 22.6],
    "Hungary": [16.0, 45.7, 22.9, 48.6],
    "Iceland": [-24.8, 63.0, -13.0, 66.8],
    "India": [68.0, 6.0, 97.5, 36.5],
    "Indonesia": [95.0, -11.2, 141.1, 6.2],
    "Ireland": [-10.8, 51.2, -5.4, 55.6],
    "Israel": [34.2, 29.4, 35.9, 33.4],
    "Italy": [6.6, 35.4, 18.6, 47.2],
    "Japan": [122.8, 24.0, 146.1, 46.1],
    "Kuwait": [46.5, 28.4, 49.1, 30.2],
    "Latvia": [20.8, 55.6, 28.3, 58.1],
    "Lithuania": [20.8, 53.8, 26.9, 56.5],
    "Malaysia": [99.6, 0.8, 119.3, 7.5],
    "Malta": [14.1, 35.7, 14.7, 36.1],
    "Mauritius": [56.8, -20.7, 58.0, -19.8],
    "Mexico": [-118.5, 14.0, -86.5, 33.2],
    "Mongolia": [87.7, 41.5, 119.9, 52.2],
    "Myanmar": [92.0, 9.5, 101.2, 28.7],
    "Netherlands": [3.1, 50.6, 7.3, 53.7],
    "New Zealand": [166.0, -47.5, 179.0, -34.0],
    "North America": [-130.0, 20.0, -60.0, 55.0],
    "Oman": [52.0, 16.5, 60.0, 26.6],
    "Pakistan": [60.8, 23.5, 77.2, 37.1],
    "Panama": [-83.1, 7.0, -77.0, 9.9],
    "Paraguay": [-62.7, -27.7, -54.2, -19.2],
    "Philippines": [116.5, 4.4, 127.0, 21.3],
    "Poland": [14.0, 49.0, 24.2, 54.9],
    "Portugal": [-9.7, 36.8, -6.1, 42.3],
    "Romania": [20.2, 43.5, 29.8, 48.4],
    "Saudi Arabia": [34.5, 15.0, 56.0, 32.5],
    "Serbia": [18.8, 42.2, 23.1, 46.3],
    "Singapore": [103.5, 1.1, 104.1, 1.6],
    "Slovakia": [16.8, 47.7, 22.7, 49.7],
    "Slovenia": [13.3, 45.4, 16.7, 46.9],
    "South Korea": [124.5, 33.0, 131.0, 39.0],
    "Spain": [-10.0, 35.8, 4.5, 44.2],
    "Sweden": [10.8, 55.0, 24.3, 69.1],
    "Switzerland": [5.8, 45.7, 10.6, 47.9],
    "Taiwan": [119.3, 21.7, 122.1, 25.5],
    "Thailand": [97.3, 5.5, 105.8, 20.6],
    "Trinidad and Tobago": [-62.1, 10.0, -60.3, 11.5],
    "Turkey": [25.5, 35.6, 45.0, 42.3],
    "United Kingdom": [-8.7, 49.8, 2.0, 60.9],
    "United States": [-126.0, 24.0, -66.0, 50.0],
    "Venezuela": [-73.4, 0.5, -59.7, 12.8],
    "Vietnam": [102.0, 8.2, 110.0, 23.5],
}


PROVIDER_ENDPOINTS = {
    "Canada": {
        "endpoint_type": "wms",
        "endpoint_url": ECCC_GEOMET_URL,
        "wms": {
            "url": ECCC_GEOMET_URL,
            "layers": ECCC_RADAR_LAYER,
            "styles": ECCC_RADAR_STYLE,
            "version": "1.3.0",
            "crs": "EPSG:4326",
            "format": "image/png",
        },
        "machine_readable": True,
        "endpoint_notes": "Active provider: MSC GeoMet WMS rain precipitation-rate layer converted from the official linear legend.",
    },
    "Europe": {
        "endpoint_type": "timestamped-image-sequence",
        "endpoint_url": OPERA_CIRRUS_LIST_IMAGES_URL,
        "machine_readable": True,
        "endpoint_notes": "Active provider: EUMETNET OPERA/CIRRUS public 5-minute maximum-reflectivity frame list decoded from the labelled dBZ palette.",
        "preferred_raw_product": {
            "name": "OPERA NIMBUS Instantaneous Rain Rate",
            "prefix": OPERA_NIMBUS_RAIN_RATE_PREFIX,
            "format": "ODIM HDF5 / Cloud-Optimized GeoTIFF",
            "source_url": OPERA_NIMBUS_PRODUCT_URL,
            "notes": "Use this product instead of CIRRUS when a reachable ORD/EWC rolling-cache URL is available; it is already rain rate and avoids reflectivity conversion.",
        },
        "reflectivity_product": {
            "name": "OPERA CIRRUS Maximum Reflectivity",
            "prefix": OPERA_CIRRUS_REFLECTIVITY_PREFIX,
            "source_url": OPERA_CIRRUS_PRODUCT_URL,
        },
    },
    "Germany": {
        "endpoint_type": "opendata-directory",
        "endpoint_url": "https://opendata.dwd.de/weather/radar/",
        "machine_readable": True,
        "endpoint_notes": "DWD open radar directories expose RADOLAN/RADVOR products; needs product-specific decoder or image mapping.",
    },
    "Japan": {
        "endpoint_type": "xyz-tile-timeseries",
        "endpoint_url": JMA_NOWCAST_TIMES_URL,
        "tile_url_template": JMA_NOWCAST_TILE_TEMPLATE,
        "machine_readable": True,
        "endpoint_notes": "Active provider: JMA nowcast time feed and hrpns raster tiles are stitched into numeric Gusty packs.",
    },
    "North America": {
        "endpoint_type": "grib2-s3-open-data",
        "endpoint_url": MRMS_BUCKET_URL,
        "machine_readable": True,
        "endpoint_notes": "Active provider: NOAA MRMS PrecipRate GRIB2 files from the public AWS Open Data bucket.",
    },
    "Netherlands": {
        "endpoint_type": "open-data-api",
        "endpoint_url": "https://api.dataplatform.knmi.nl/open-data/v1/datasets/radar_reflectivity_composites/versions/2.0/files",
        "machine_readable": True,
        "endpoint_notes": "KNMI Data Platform radar reflectivity composites. May require API key/header depending on access policy.",
    },
    "Spain": {
        "endpoint_type": "open-data-api",
        "endpoint_url": "https://opendata.aemet.es/opendata/api/red/radar/nacional",
        "machine_readable": True,
        "endpoint_notes": "AEMET OpenData endpoint normally requires an API key.",
    },
    "Poland": {
        "endpoint_type": "open-data-directory",
        "endpoint_url": "https://danepubliczne.imgw.pl/datastore",
        "machine_readable": True,
        "endpoint_notes": "IMGW public datastore; exact radar product path and palette mapping still required.",
    },
    "Singapore": {
        "endpoint_type": "timestamped-image-sequence",
        "endpoint_url": "https://www.weather.gov.sg/files/rainarea/50km/v2/",
        "image_url_template": "https://www.weather.gov.sg/files/rainarea/50km/v2/dpsri_70km_{yyyymmddHHMM}0000dBR.dpsri.png",
        "alternate_image_url_templates": [
            "https://www.weather.gov.sg/files/rainarea/240km/v2/dpsri_240km_{yyyymmddHHMM}0000dBR.dpsri.png",
            "https://www.weather.gov.sg/files/rainarea/480km/v2/dpsri_480km_{yyyymmddHHMM}0000dBR.dpsri.png",
        ],
        "machine_readable": True,
        "endpoint_notes": "MSS page exposes timestamped rain-area PNGs and a legend image. Needs timestamp discovery and color mapping before activation.",
    },
    "Taiwan": {
        "endpoint_type": "open-data-api",
        "endpoint_url": "https://opendata.cwa.gov.tw/",
        "machine_readable": True,
        "endpoint_notes": "CWA open data portal. Radar product selection and API authorization may be required.",
    },
    "United States": {
        "endpoint_type": "georeferenced-image",
        "endpoint_url": f"{IEM_CURRENT_URL}/n0q_0.png",
        "world_file_url": f"{IEM_CURRENT_URL}/n0q_0.wld",
        "machine_readable": True,
        "endpoint_notes": "Already active in this fetcher through IEM NEXRAD current raster frames.",
    },
}


ADDITIONAL_PROVIDER_URLS = {
    "Bangladesh": ["https://met.baf.mil.bd/"],
    "Brazil": [
        "http://www.starnet.iag.usp.br/chuvaonline/radar_xpol.php",
        "http://www.ipmet.unesp.br/2imagemRadar.php",
        "http://www.simepar.br/prognozweb/simepar/radar_msc",
        "http://alertadecheias.inea.rj.gov.br/alertadecheias/radar.html",
        "http://www.funceme.br/app/radar/animacao/quixeramobim",
        "http://sigsc.sc.gov.br",
        "http://alertario.rio.rj.gov.br/radar-meteorologico-do-sumare/imgens-recentes/",
        "http://www.cemaden.gov.br/mapainterativo/",
        "https://wp.ufpel.edu.br/cppmet/radar/",
        "https://barbadosweather.org/",
        "http://www.radar.niteroi.rj.gov.br/",
    ],
    "Czechia": ["http://radar.bourky.cz/"],
    "Estonia": ["https://keskkonnaportaal.ee"],
    "France": [
        "http://www.meteo60.fr/radars-precipitations-pluie-france.php",
        "https://aviation.meteo.fr/",
        "https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=307&id_rubrique=34",
    ],
    "Germany": ["http://iras.skywarn.de/public/"],
    "Greece": ["https://www.metoc.navy.mil/fwcn/animate.html?icao=kqnc&type=BaseZH120"],
    "India": ["http://ddgmui.imd.gov.in/radar/leaflet-map-csv-master/mosaic.php"],
    "Israel": ["http://mekorotapp.co.il/manager/radars/radar"],
    "Italy": [
        "http://93.62.155.214",
        "http://www.arpae.it/sim/?osservazioni_e_dati/radar",
        "http://www.arpa.veneto.it/bollettini/meteo/radar/radar.php",
        "https://www.meteo.fvg.it/",
        "https://meteo.provincia.bz.it/radar-meteorologico.asp",
    ],
    "Mexico": [
        "https://guanajuato.gob.mx",
        "http://astro.iam.udg.mx/radar/",
        "https://aplicaciones.sacmex.cdmx.gob.mx/radar-meteorologico/",
    ],
    "Panama": ["http://www.pancanal.com/radar-meteorologico/"],
    "Spain": ["https://www.euskalmet.euskadi.eus"],
    "Sweden": ["http://www.smhi.se/"],
}


def radar_provider_slug(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


configured_regions = {item["region"] for item in CONFIGURABLE_RADAR_REGIONS}
for country, attribution, source_url in ADDITIONAL_CONFIGURABLE_REGION_SOURCES:
    if country in configured_regions:
        continue
    CONFIGURABLE_RADAR_REGIONS.append(
        {
            "id": f"radar_{radar_provider_slug(country)}_configured",
            "name": f"{country} Radar Intensity",
            "region": country,
            "attribution": attribution,
            "source_url": source_url,
            "notes": "Configure an approved image or WMS endpoint plus index_values or color_values to publish numeric radar packs.",
        }
    )
    configured_regions.add(country)


def provider_endpoint(region: str) -> Dict[str, Any]:
    endpoint = PROVIDER_ENDPOINTS.get(region)
    if endpoint:
        return endpoint
    return {
        "endpoint_type": "provider-page",
        "endpoint_url": None,
        "machine_readable": False,
        "endpoint_notes": "Only a provider radar page/source URL is known. Exact image, WMS, API, or tile endpoint still needs discovery.",
    }


def candidate_urls(region: str, provider_url: str) -> List[str]:
    urls = [provider_url]
    for url in ADDITIONAL_PROVIDER_URLS.get(region, []):
        if url not in urls:
            urls.append(url)
    return urls

TILE_SIZE = 20
PACK_LAT_SIZE = 60
PACK_LON_SIZE = 80
OVERVIEW_KEY_SUFFIX = "_60"
OVERVIEW_SAMPLE_STEP = 4
def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


RADAR_TARGET_DEGREES = env_float("RADAR_TARGET_DEGREES", 0.025)
MRMS_TARGET_DEGREES = env_float("MRMS_TARGET_DEGREES", max(RADAR_TARGET_DEGREES, 0.05))
RADAR_MIN_VALUE = env_float("RADAR_MIN_VALUE", 0)
RADAR_VALUE_PARAMETER = "rain_rate"
RADAR_VALUE_UNITS = "mm/h"
RADAR_SOURCE_REFERENCE = "official and public meteorological radar provider references"
RADAR_COLOR_STOPS = [
    {"value": 0.1, "label": "trace"},
    {"value": 1.0, "label": "light"},
    {"value": 4.0, "label": "moderate"},
    {"value": 16.0, "label": "heavy"},
    {"value": 32.0, "label": "very_heavy"},
    {"value": 64.0, "label": "extreme"},
]
RADAR_VECTOR_BANDS = [
    {"min": 0.1, "max": 1.0, "class": "trace"},
    {"min": 1.0, "max": 4.0, "class": "light"},
    {"min": 4.0, "max": 16.0, "class": "moderate"},
    {"min": 16.0, "max": 32.0, "class": "heavy"},
    {"min": 32.0, "max": 64.0, "class": "very_heavy"},
    {"min": 64.0, "max": None, "class": "extreme"},
]

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
VECTOR_TILE_FORMAT = {
    "name": "gusty-vector-json",
    "version": 1,
    "extension": "gjson",
    "encoding": "utf-8-json",
    "geometry": "polygon",
    "coordinates": "lonlat",
    "parameter": RADAR_VALUE_PARAMETER,
    "units": RADAR_VALUE_UNITS,
    "bands": RADAR_VECTOR_BANDS,
}
VECTOR_PACK_FORMAT = {
    "name": "gusty-vector-pack",
    "version": 1,
    "extension": "gpack",
    "contains": VECTOR_TILE_FORMAT["extension"],
    "lat_size": PACK_LAT_SIZE,
    "lon_size": PACK_LON_SIZE,
    "offset_base": "start_of_data_section",
}

MAGIC = b"GSTY"
VERSION = 1
ENCODING_INT16_QUANTIZED = 1
INT16_MIN_VALUE = -32767
INT16_MAX_VALUE = 32767
PACK_MAGIC = b"GPAK"
PACK_VERSION = 1
PARAMETER_RAIN_RATE = 0


@dataclass
class RadarGrid:
    provider_id: str
    name: str
    values: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    ref_time: Optional[str]
    source_url: str
    attribution: str
    region: str
    units: str = RADAR_VALUE_UNITS
    source_units: Optional[str] = None
    source_parameter: Optional[str] = None
    target_degrees: float = RADAR_TARGET_DEGREES
    status: str = "ok"
    error: Optional[str] = None


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def session() -> requests.Session:
    client = requests.Session()
    client.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "image/png, text/plain, application/json, */*",
        }
    )
    return client


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)


def clean_provider_dir(provider_id: str) -> str:
    provider_dir = os.path.join(RADAR_DIR, provider_id)
    if os.path.exists(provider_dir):
        shutil.rmtree(provider_dir)
    os.makedirs(provider_dir, exist_ok=True)
    return provider_dir


def clean_vector_provider_dir(provider_id: str) -> str:
    provider_dir = os.path.join(RADAR_VECTOR_DIR, provider_id)
    if os.path.exists(provider_dir):
        shutil.rmtree(provider_dir)
    os.makedirs(provider_dir, exist_ok=True)
    return provider_dir


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


def pack_filename(provider_id: str, frame: int, lat_start: int, lon_start: int) -> str:
    lat_label, lon_label = coordinate_label(lat_start, lon_start)
    return f"{provider_id}_{frame}h_{lat_label}_{lon_label}.{PACK_FORMAT['extension']}"


def overview_tile_key(lat_start: int, lon_start: int) -> str:
    return f"{tile_key(lat_start, lon_start)}{OVERVIEW_KEY_SUFFIX}"


def vector_band_index(value: float) -> Optional[int]:
    if value < RADAR_VECTOR_BANDS[0]["min"]:
        return None
    for index, band in enumerate(RADAR_VECTOR_BANDS):
        max_value = band["max"]
        if value >= band["min"] and (max_value is None or value < max_value):
            return index
    return None


def vector_tile_bytes(
    lon_vals: np.ndarray,
    lat_vals: np.ndarray,
    dx: float,
    dy: float,
    values: np.ndarray,
) -> Optional[bytes]:
    if values.size == 0 or float(np.nanmax(values)) < RADAR_VECTOR_BANDS[0]["min"]:
        return None

    features = []
    lon_half = abs(dx) / 2.0
    lat_half = abs(dy) / 2.0
    for row_index, lat in enumerate(lat_vals):
        current_band: Optional[int] = None
        run_start: Optional[int] = None
        run_values: List[float] = []
        row_values = values[row_index]
        for col_index in range(len(lon_vals) + 1):
            if col_index < len(lon_vals):
                value = float(row_values[col_index])
                band_index = vector_band_index(value)
            else:
                value = 0.0
                band_index = None

            if band_index == current_band:
                if band_index is not None:
                    run_values.append(value)
                continue

            if current_band is not None and run_start is not None and run_values:
                lon_min = float(lon_vals[run_start] - lon_half)
                lon_max = float(lon_vals[col_index - 1] + lon_half)
                lat_min = float(lat - lat_half)
                lat_max = float(lat + lat_half)
                band = RADAR_VECTOR_BANDS[current_band]
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "class": band["class"],
                            "min": band["min"],
                            "max": band["max"],
                            "value": round(max(run_values), 1),
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [round(lon_min, 5), round(lat_min, 5)],
                                    [round(lon_max, 5), round(lat_min, 5)],
                                    [round(lon_max, 5), round(lat_max, 5)],
                                    [round(lon_min, 5), round(lat_max, 5)],
                                    [round(lon_min, 5), round(lat_min, 5)],
                                ]
                            ],
                        },
                    }
                )

            current_band = band_index
            run_start = col_index if band_index is not None else None
            run_values = [value] if band_index is not None else []

    if not features:
        return None

    payload = {
        "type": "FeatureCollection",
        "parameter": RADAR_VALUE_PARAMETER,
        "units": RADAR_VALUE_UNITS,
        "features": features,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def quantize_int16(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
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
    flattened = np.round(np.nan_to_num(values, nan=0.0).astype(np.float32), 1).flatten()
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
    output = bytearray(header)
    output.extend(quantized.tobytes())
    return bytes(output)


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

    with open(pack_path, "wb") as handle:
        handle.write(header)
        handle.write(data)


def parse_last_modified(headers: Dict[str, str]) -> Optional[str]:
    value = headers.get("Last-Modified")
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()
    except (TypeError, ValueError):
        return None


def get_bytes(client: requests.Session, url: str) -> Tuple[bytes, Optional[str]]:
    response = client.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.content, parse_last_modified(response.headers)


def get_text(client: requests.Session, url: str) -> Tuple[str, Optional[str]]:
    response = client.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text, parse_last_modified(response.headers)


def read_world_file(text: str) -> Tuple[float, float, float, float]:
    values = [float(line.strip()) for line in text.splitlines() if line.strip()]
    if len(values) < 6:
        raise ValueError("World file must contain at least 6 numeric rows")
    x_pixel_size = values[0]
    y_pixel_size = values[3]
    x_origin = values[4]
    y_origin = values[5]
    return x_pixel_size, y_pixel_size, x_origin, y_origin


def n0q_indices_to_dbz(index_values: np.ndarray) -> np.ndarray:
    dbz = -32.0 + (index_values.astype(np.float32) - 1.0) * 0.5
    dbz = np.where(index_values <= 1, 0.0, dbz)
    return np.clip(dbz, 0.0, 90.0).astype(np.float32)


def dbz_to_rain_rate(dbz: np.ndarray) -> np.ndarray:
    z = np.power(10.0, dbz.astype(np.float32) / 10.0)
    rain_rate = np.power(z / 200.0, 1.0 / 1.6)
    rain_rate = np.where(dbz <= 0, 0.0, rain_rate)
    rain_rate = np.where(rain_rate < RADAR_MIN_VALUE, 0.0, rain_rate)
    return np.clip(rain_rate, 0.0, 250.0).astype(np.float32)


def standardize_values(values: np.ndarray, units: Optional[str]) -> Tuple[np.ndarray, Optional[str], Optional[str]]:
    source_units = units or RADAR_VALUE_UNITS
    normalized_units = source_units.strip().lower().replace(" ", "")
    if normalized_units in {"dbz", "db"}:
        return dbz_to_rain_rate(values), source_units, "reflectivity"
    if normalized_units in {"mm/h", "mm/hr", "mmh", "mmperhour", "mm/hour"}:
        rain_rate = np.where(values < RADAR_MIN_VALUE, 0.0, values)
        return rain_rate.astype(np.float32), source_units, "rain_rate"
    raise ValueError(f"Unsupported radar units for unified rain-rate output: {source_units}")


def parse_index_values(value: str) -> np.ndarray:
    mapping = np.zeros(256, dtype=np.float32)
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) == 256 and all(":" not in item for item in parts):
        return np.array([float(item) for item in parts], dtype=np.float32)
    for item in parts:
        if ":" not in item:
            raise ValueError("Palette mapping must be 256 values or comma-separated index:value pairs")
        index_text, value_text = item.split(":", 1)
        index = int(index_text.strip())
        if index < 0 or index > 255:
            raise ValueError(f"Palette index out of range: {index}")
        mapping[index] = float(value_text.strip())
    return mapping


def index_mapping_from_config(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        mapping = np.zeros(256, dtype=np.float32)
        for index_text, mapped_value in value.items():
            index = int(index_text)
            if index < 0 or index > 255:
                raise ValueError(f"Palette index out of range: {index}")
            mapping[index] = float(mapped_value)
        return mapping
    if isinstance(value, list):
        if len(value) != 256:
            raise ValueError("index_values list must contain exactly 256 values")
        return np.array([float(item) for item in value], dtype=np.float32)
    return parse_index_values(str(value))


def indices_to_configured_values(index_values: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    values = mapping[index_values.astype(np.uint8)]
    return np.where(values < RADAR_MIN_VALUE, 0.0, values).astype(np.float32)


def normalize_hex_color(value: str) -> Tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"Invalid hex color: {value}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def rgba_to_configured_values(image: Image.Image, color_values: Dict[str, Any]) -> np.ndarray:
    mapping = {
        normalize_hex_color(color): float(mapped_value)
        for color, mapped_value in color_values.items()
    }
    rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
    output = np.zeros(rgba.shape[:2], dtype=np.float32)
    for color, mapped_value in mapping.items():
        mask = (
            (rgba[:, :, 0] == color[0])
            & (rgba[:, :, 1] == color[1])
            & (rgba[:, :, 2] == color[2])
            & (rgba[:, :, 3] > 0)
        )
        output[mask] = mapped_value
    return np.where(output < RADAR_MIN_VALUE, 0.0, output).astype(np.float32)


def rgb_to_configured_values(image_bytes: bytes, color_values: Dict[str, Any]) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return rgba_to_configured_values(image, color_values)


def image_to_palette_indices(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        indexed = image.convert("P") if image.mode != "P" else image
        return np.array(indexed, dtype=np.uint8)


def sample_grid(
    values: np.ndarray,
    x_pixel_size: float,
    y_pixel_size: float,
    x_origin: float,
    y_origin: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    native_resolution = max(abs(x_pixel_size), abs(y_pixel_size))
    sample_step = max(1, int(round(RADAR_TARGET_DEGREES / native_resolution)))
    sampled = values[::sample_step, ::sample_step]
    rows = np.arange(0, values.shape[0], sample_step, dtype=np.float64)
    cols = np.arange(0, values.shape[1], sample_step, dtype=np.float64)
    longitudes = x_origin + cols * x_pixel_size
    latitudes = y_origin + rows * y_pixel_size
    if latitudes[0] < latitudes[-1]:
        sampled = sampled[::-1, :]
        latitudes = latitudes[::-1]
    return sampled, latitudes, longitudes


def sample_lonlat_grid(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    target_degrees: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    lat_resolution = abs(float(np.nanmedian(np.diff(latitudes)))) if len(latitudes) > 1 else target_degrees
    lon_resolution = abs(float(np.nanmedian(np.diff(longitudes)))) if len(longitudes) > 1 else target_degrees
    native_resolution = max(min(lat_resolution, lon_resolution), 0.0001)
    sample_step = max(1, int(round(target_degrees / native_resolution)))
    sampled = values[::sample_step, ::sample_step]
    sampled_latitudes = latitudes[::sample_step]
    sampled_longitudes = longitudes[::sample_step]
    if sampled_latitudes[0] < sampled_latitudes[-1]:
        sampled = sampled[::-1, :]
        sampled_latitudes = sampled_latitudes[::-1]
    if sampled_longitudes[0] > sampled_longitudes[-1]:
        order = np.argsort(sampled_longitudes)
        sampled = sampled[:, order]
        sampled_longitudes = sampled_longitudes[order]
    return sampled, sampled_latitudes, sampled_longitudes


def parse_mrms_time_from_key(key: str) -> Optional[str]:
    match = re.search(r"_(\d{8})-(\d{6})\.grib2\.gz$", key)
    if not match:
        return None
    parsed = dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    return parsed.replace(tzinfo=dt.timezone.utc).isoformat()


def list_mrms_preciprate_keys(client: requests.Session, day: dt.date) -> List[Tuple[str, str]]:
    prefix = f"{MRMS_PRODUCT_PREFIX}/{day:%Y%m%d}/"
    params = urlencode({"list-type": "2", "prefix": prefix, "max-keys": "1000"})
    text, _ = get_text(client, f"{MRMS_BUCKET_URL}/?{params}")
    root = ET.fromstring(text)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = []
    for item in root.findall("s3:Contents", namespace):
        key = item.findtext("s3:Key", default="", namespaces=namespace)
        modified = item.findtext("s3:LastModified", default="", namespaces=namespace)
        if key.endswith(".grib2.gz"):
            keys.append((modified, key))
    return keys


def latest_mrms_preciprate_key(client: requests.Session) -> Tuple[str, Optional[str]]:
    today = dt.datetime.now(dt.timezone.utc).date()
    keys: List[Tuple[str, str]] = []
    for offset in (0, 1):
        keys.extend(list_mrms_preciprate_keys(client, today - dt.timedelta(days=offset)))
        if keys:
            break
    if not keys:
        raise ValueError("No current MRMS PrecipRate GRIB2 files found")
    modified, key = max(keys, key=lambda item: item[0])
    return key, modified or None


def fetch_mrms_precip_rate(client: requests.Session) -> RadarGrid:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("NOAA MRMS decoding requires xarray and cfgrib") from exc

    key, modified = latest_mrms_preciprate_key(client)
    grib_gz_url = f"{MRMS_BUCKET_URL}/{key}"
    grib_gz_bytes, http_time = get_bytes(client, grib_gz_url)
    grib_bytes = gzip.decompress(grib_gz_bytes)
    with tempfile.NamedTemporaryFile(suffix=".grib2") as handle:
        handle.write(grib_bytes)
        handle.flush()
        dataset = xr.open_dataset(handle.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
        data_var = next(iter(dataset.data_vars))
        data = dataset[data_var]
        values = data.values.astype(np.float32)
        latitudes = dataset["latitude"].values.astype(np.float64)
        longitudes = dataset["longitude"].values.astype(np.float64)
        valid_time = dataset.coords.get("valid_time")
        ref_time = None
        if valid_time is not None:
            ref_time = np.datetime_as_string(valid_time.values, unit="s") + "+00:00"

    longitudes = np.where(longitudes > 180.0, longitudes - 360.0, longitudes)
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    values = np.clip(values, 0.0, 250.0).astype(np.float32)
    values, latitudes, longitudes = sample_lonlat_grid(values, latitudes, longitudes, MRMS_TARGET_DEGREES)
    return RadarGrid(
        provider_id="radar_north_america_mrms",
        name="North America MRMS Rain Rate",
        values=values,
        latitudes=latitudes,
        longitudes=longitudes,
        ref_time=ref_time or parse_mrms_time_from_key(key) or http_time or modified,
        source_url=MRMS_DOCS_URL,
        attribution="NOAA MRMS / AWS Open Data",
        region="North America",
        units=RADAR_VALUE_UNITS,
        source_units=RADAR_VALUE_UNITS,
        source_parameter=RADAR_VALUE_PARAMETER,
        target_degrees=MRMS_TARGET_DEGREES,
    )


def fetch_us_iem_nexrad(client: requests.Session) -> RadarGrid:
    image_url = f"{IEM_CURRENT_URL}/n0q_0.png"
    world_url = f"{IEM_CURRENT_URL}/n0q_0.wld"
    image_bytes, image_time = get_bytes(client, image_url)
    world_text, world_time = get_text(client, world_url)
    x_pixel_size, y_pixel_size, x_origin, y_origin = read_world_file(world_text)
    indices = image_to_palette_indices(image_bytes)
    dbz = n0q_indices_to_dbz(indices)
    rain_rate, source_units, source_parameter = standardize_values(dbz, "dBZ")
    values, latitudes, longitudes = sample_grid(rain_rate, x_pixel_size, y_pixel_size, x_origin, y_origin)
    return RadarGrid(
        provider_id="radar_us_iem_nexrad",
        name="US NEXRAD Rain Rate",
        values=values,
        latitudes=latitudes,
        longitudes=longitudes,
        ref_time=image_time or world_time,
        source_url=IEM_N0Q_DOCS_URL,
        attribution="Iowa Environmental Mesonet / NOAA NEXRAD",
        region="United States",
        source_units=source_units,
        source_parameter=source_parameter,
    )


def lon_to_tile_x(lon: float, zoom: int) -> int:
    return int(math.floor((lon + 180.0) / 360.0 * (2 ** zoom)))


def lat_to_tile_y(lat: float, zoom: int) -> int:
    lat_rad = math.radians(lat)
    return int(
        math.floor(
            (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi)
            / 2.0
            * (2 ** zoom)
        )
    )


def tile_pixel_to_lonlat(x: int, y: int, zoom: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
    n = float(2 ** zoom * size)
    cols = np.arange(x * size, (x + 1) * size, dtype=np.float64) + 0.5
    rows = np.arange(y * size, (y + 1) * size, dtype=np.float64) + 0.5
    longitudes = cols / n * 360.0 - 180.0
    mercator = math.pi * (1.0 - 2.0 * rows / n)
    latitudes = np.degrees(np.arctan(np.sinh(mercator)))
    return longitudes, latitudes


def parse_jma_time(value: str) -> str:
    parsed = dt.datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    return parsed.isoformat()


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

ECCC_RADAR_COLOR_RAMP = [
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

OPERA_CIRRUS_DBZ_RAMP = [
    (50.0, "#beffff"),
    (45.0, "#fa78ff"),
    (40.0, "#ff503c"),
    (34.0, "#ff9632"),
    (30.0, "#ffcd14"),
    (24.0, "#f0f014"),
    (18.0, "#8ce614"),
    (12.0, "#05cdaa"),
    (8.0, "#0ab9af"),
    (0.0, "#0a9bb4"),
    (-6.0, "#0a82c8"),
]


def rgba_to_color_ramp_values(image: Image.Image, ramp: List[Tuple[float, str]]) -> np.ndarray:
    rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
    output = np.zeros(rgba.shape[:2], dtype=np.float32)
    mask = rgba[:, :, 3] > 0
    if not np.any(mask):
        return output

    colors = np.array([normalize_hex_color(color) for _, color in ramp], dtype=np.int16)
    values = np.array([value for value, _ in ramp], dtype=np.float32)
    pixels = rgba[:, :, :3][mask].astype(np.int16)
    distances = np.sum((pixels[:, None, :] - colors[None, :, :]) ** 2, axis=2)
    output[mask] = values[np.argmin(distances, axis=1)]
    return np.where(output < RADAR_MIN_VALUE, 0.0, output).astype(np.float32)


def rgba_to_threshold_ramp_values(
    image: Image.Image,
    ramp: List[Tuple[float, str]],
    max_color_distance: int = 900,
    exclude_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
    output = np.zeros(rgba.shape[:2], dtype=np.float32)
    mask = rgba[:, :, 3] > 0
    if exclude_boxes:
        for left, top, right, bottom in exclude_boxes:
            mask[top:bottom, left:right] = False
    if not np.any(mask):
        return output

    colors = np.array([normalize_hex_color(color) for _, color in ramp], dtype=np.int16)
    values = np.array([value for value, _ in ramp], dtype=np.float32)
    pixels = rgba[:, :, :3][mask].astype(np.int16)
    distances = np.sum((pixels[:, None, :] - colors[None, :, :]) ** 2, axis=2)
    closest = np.argmin(distances, axis=1)
    accepted = distances[np.arange(len(closest)), closest] <= max_color_distance
    mapped = np.zeros(len(closest), dtype=np.float32)
    mapped[accepted] = values[closest[accepted]]
    output[mask] = mapped
    return output.astype(np.float32)


def epoch_millis_to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return dt.datetime.fromtimestamp(float(value) / 1000.0, tz=dt.timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def eccc_radar_wms_url(bounds: Tuple[float, float, float, float]) -> str:
    west, south, east, north = bounds
    width = int(os.getenv("CANADA_RADAR_WIDTH", "1800"))
    height = int(os.getenv("CANADA_RADAR_HEIGHT", "900"))
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": ECCC_RADAR_LAYER,
        "STYLES": ECCC_RADAR_STYLE,
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
        "CRS": "EPSG:4326",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "BBOX": f"{south},{west},{north},{east}",
    }
    return f"{ECCC_GEOMET_URL}?{urlencode(params)}"


def fetch_canada_geomet_radar(client: requests.Session) -> RadarGrid:
    bounds = REGION_BOUNDS["Canada"]
    west, south, east, north = bounds
    image_url = eccc_radar_wms_url(bounds)
    image_bytes, image_time = get_bytes(client, image_url)
    with Image.open(io.BytesIO(image_bytes)) as image:
        values = rgba_to_color_ramp_values(image, ECCC_RADAR_COLOR_RAMP)
    height, width = values.shape
    x_pixel_size = (east - west) / max(width - 1, 1)
    y_pixel_size = (south - north) / max(height - 1, 1)
    values, latitudes, longitudes = sample_grid(values, x_pixel_size, y_pixel_size, west, north)
    return RadarGrid(
        provider_id="radar_canada_configured",
        name="Canada Radar Intensity",
        values=values,
        latitudes=latitudes,
        longitudes=longitudes,
        ref_time=image_time,
        source_url="https://eccc-msc.github.io/open-data/msc-data/obs_radar/readme_radar_geomet_en/",
        attribution="Environment and Climate Change Canada / MSC GeoMet",
        region="Canada",
        units="mm/h",
        source_units="mm/h",
        source_parameter=RADAR_VALUE_PARAMETER,
    )


def latest_opera_cirrus_frame(client: requests.Session) -> Tuple[str, Optional[str]]:
    response = client.get(OPERA_CIRRUS_LIST_IMAGES_URL, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    images = payload.get("images") or []
    if not images:
        raise ValueError("OPERA/CIRRUS frame list is empty")
    latest = max(images, key=lambda item: item.get("epoch") or 0)
    url = latest.get("url")
    if not url:
        raise ValueError("Latest OPERA/CIRRUS frame has no URL")
    ref_time = epoch_millis_to_iso(latest.get("epoch")) or epoch_millis_to_iso(payload.get("lastModified"))
    return url, ref_time


def fetch_opera_cirrus_radar(client: requests.Session) -> RadarGrid:
    bounds = REGION_BOUNDS["Europe"]
    west, south, east, north = bounds
    image_url, ref_time = latest_opera_cirrus_frame(client)
    image_bytes, image_time = get_bytes(client, image_url)
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        dbz = rgba_to_threshold_ramp_values(
            image,
            OPERA_CIRRUS_DBZ_RAMP,
            exclude_boxes=[
                (830, 0, width, 370),
                (810, max(0, height - 130), width, height),
            ],
        )
    rain_rate, source_units, source_parameter = standardize_values(dbz, "dBZ")
    x_pixel_size = (east - west) / max(width - 1, 1)
    y_pixel_size = (south - north) / max(height - 1, 1)
    values, latitudes, longitudes = sample_grid(rain_rate, x_pixel_size, y_pixel_size, west, north)
    return RadarGrid(
        provider_id="radar_europe_opera_cirrus",
        name="Europe OPERA/CIRRUS Rain Rate",
        values=values,
        latitudes=latitudes,
        longitudes=longitudes,
        ref_time=ref_time or image_time,
        source_url=OPERA_CIRRUS_ANIMATOR_URL,
        attribution="EUMETNET OPERA / Finnish Meteorological Institute",
        region="Europe",
        source_units=source_units,
        source_parameter=source_parameter,
    )


def fetch_jma_nowcast(client: requests.Session) -> RadarGrid:
    response = client.get(JMA_NOWCAST_TIMES_URL, timeout=TIMEOUT)
    response.raise_for_status()
    times = response.json()
    if not times:
        raise ValueError("JMA nowcast time feed is empty")
    frame = times[0]
    base_time = frame["basetime"]
    valid_time = frame["validtime"]
    zoom = int(os.getenv("JMA_RADAR_ZOOM", "6"))
    bounds = REGION_BOUNDS["Japan"]
    west, south, east, north = bounds
    x_start = lon_to_tile_x(west, zoom)
    x_end = lon_to_tile_x(east, zoom)
    y_start = lat_to_tile_y(north, zoom)
    y_end = lat_to_tile_y(south, zoom)

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
    lon_values = []
    for tile_x in range(x_start, x_end + 1):
        lon_values.append(tile_pixel_to_lonlat(tile_x, y_start, zoom, 256)[0])
    longitudes = np.concatenate(lon_values)
    lat_values = []
    for tile_y in range(y_start, y_end + 1):
        lat_values.append(tile_pixel_to_lonlat(x_start, tile_y, zoom, 256)[1])
    latitudes = np.concatenate(lat_values)

    lon_mask = (longitudes >= west) & (longitudes <= east)
    lat_mask = (latitudes >= south) & (latitudes <= north)
    values = values[np.ix_(lat_mask, lon_mask)]
    latitudes = latitudes[lat_mask]
    longitudes = longitudes[lon_mask]
    values, latitudes, longitudes = sample_grid(
        values,
        longitudes[1] - longitudes[0],
        latitudes[1] - latitudes[0],
        longitudes[0],
        latitudes[0],
    )
    return RadarGrid(
        provider_id="radar_japan_configured",
        name="Japan Radar Intensity",
        values=values,
        latitudes=latitudes,
        longitudes=longitudes,
        ref_time=parse_jma_time(valid_time),
        source_url="https://www.jma.go.jp/jp/radnowc/index.html",
        attribution="Japan Meteorological Agency",
        region="Japan",
        units="mm/h",
        source_units="mm/h",
        source_parameter=RADAR_VALUE_PARAMETER,
    )


def parse_bounds(value: str) -> Tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Bounds must be 'west,south,east,north'")
    return parts[0], parts[1], parts[2], parts[3]


def bounds_from_config(value: Any) -> Tuple[float, float, float, float]:
    if value is None:
        raise ValueError("bounds is required unless region has a built-in bounds entry")
    if isinstance(value, str):
        return parse_bounds(value)
    if isinstance(value, list) and len(value) == 4:
        return float(value[0]), float(value[1]), float(value[2]), float(value[3])
    raise ValueError("bounds must be 'west,south,east,north' or a four-item list")


def wms_image_url(config: Dict[str, Any], bounds: Tuple[float, float, float, float]) -> str:
    wms = config.get("wms") or {}
    service_url = wms.get("url") or config.get("wms_url")
    layers = wms.get("layers") or config.get("wms_layers")
    if not service_url or not layers:
        raise ValueError("WMS providers require wms.url and wms.layers")
    version = str(wms.get("version") or config.get("wms_version") or "1.1.1")
    crs_key = "SRS" if version.startswith("1.1") else "CRS"
    crs = wms.get("crs") or config.get("wms_crs") or "EPSG:4326"
    params = {
        "SERVICE": "WMS",
        "VERSION": version,
        "REQUEST": "GetMap",
        "LAYERS": layers,
        "STYLES": wms.get("styles") or config.get("wms_styles") or "",
        "FORMAT": wms.get("format") or config.get("wms_format") or "image/png",
        "TRANSPARENT": str(wms.get("transparent", True)).lower(),
        crs_key: crs,
        "WIDTH": str(wms.get("width") or config.get("width") or 1200),
        "HEIGHT": str(wms.get("height") or config.get("height") or 1200),
        "BBOX": ",".join(str(part) for part in bounds),
    }
    time_value = wms.get("time") or config.get("time")
    if time_value:
        params["TIME"] = time_value
    return f"{service_url}?{urlencode(params)}"


def configured_image_values(image_bytes: bytes, config: Dict[str, Any]) -> np.ndarray:
    if "index_values" in config:
        indices = image_to_palette_indices(image_bytes)
        return indices_to_configured_values(indices, index_mapping_from_config(config["index_values"]))
    if "color_values" in config:
        return rgb_to_configured_values(image_bytes, config["color_values"])
    raise ValueError("Configured providers require index_values or color_values")


def fetch_configured_radar(client: requests.Session, config: Dict[str, Any]) -> RadarGrid:
    provider_id = config["id"]
    bounds_value = config.get("bounds")
    if bounds_value is None:
        bounds_value = REGION_BOUNDS.get(config.get("region", ""))
    bounds = bounds_from_config(bounds_value)
    image_url = config.get("image_url") or wms_image_url(config, bounds)
    west, south, east, north = bounds
    image_bytes, image_time = get_bytes(client, image_url)
    source_units = config.get("units", RADAR_VALUE_UNITS)
    values, standardized_source_units, source_parameter = standardize_values(
        configured_image_values(image_bytes, config),
        source_units,
    )
    height, width = values.shape
    x_pixel_size = (east - west) / max(width - 1, 1)
    y_pixel_size = (south - north) / max(height - 1, 1)
    values, latitudes, longitudes = sample_grid(values, x_pixel_size, y_pixel_size, west, north)
    return RadarGrid(
        provider_id=provider_id,
        name=config.get("name", provider_id),
        values=values,
        latitudes=latitudes,
        longitudes=longitudes,
        ref_time=config.get("ref_time") or image_time,
        source_url=config.get("source_url") or image_url,
        attribution=config.get("attribution", provider_id),
        region=config.get("region", "Configured radar"),
        units=RADAR_VALUE_UNITS,
        source_units=standardized_source_units,
        source_parameter=source_parameter,
    )


def fetch_configured_uk_radar(client: requests.Session) -> Optional[RadarGrid]:
    image_url = os.getenv("UK_RADAR_IMAGE_URL")
    if not image_url:
        return None
    config = {
        "id": "radar_uk_configured",
        "name": "UK Radar Reflectivity",
        "region": "United Kingdom",
        "image_url": image_url,
        "bounds": os.getenv("UK_RADAR_IMAGE_BOUNDS"),
        "index_values": os.getenv("UK_RADAR_INDEX_VALUES"),
        "units": os.getenv("UK_RADAR_UNITS") or RADAR_VALUE_UNITS,
        "ref_time": os.getenv("UK_RADAR_REF_TIME"),
        "source_url": os.getenv("UK_RADAR_SOURCE_URL") or MET_OFFICE_PUBLIC_RADAR_URL,
        "attribution": os.getenv("UK_RADAR_ATTRIBUTION") or "Met Office",
    }
    return fetch_configured_radar(client, config)


def tile_range(values: Iterable[float], tile_size: int) -> range:
    numeric = list(values)
    start = math.floor(min(numeric) / tile_size) * tile_size
    end = math.floor(max(numeric) / tile_size) * tile_size
    return range(start, end + tile_size, tile_size)


def build_overview_tile(
    lon_vals: np.ndarray,
    lat_vals: np.ndarray,
    values: np.ndarray,
) -> Optional[bytes]:
    overview_values = values[::OVERVIEW_SAMPLE_STEP, ::OVERVIEW_SAMPLE_STEP]
    overview_lat = lat_vals[::OVERVIEW_SAMPLE_STEP]
    overview_lon = lon_vals[::OVERVIEW_SAMPLE_STEP]
    if overview_values.size == 0 or overview_lat.size == 0 or overview_lon.size == 0:
        return None
    dy = (overview_lat[-1] - overview_lat[0]) / (len(overview_lat) - 1) if len(overview_lat) > 1 else 1.0
    dx = (overview_lon[-1] - overview_lon[0]) / (len(overview_lon) - 1) if len(overview_lon) > 1 else 1.0
    return build_binary_tile_bytes(overview_lon, overview_lat, dx, dy, overview_values)


def generate_tiles(grid: RadarGrid) -> Dict[str, Any]:
    provider_dir = clean_provider_dir(grid.provider_id)
    vector_provider_dir = clean_vector_provider_dir(grid.provider_id)
    pack_entries: Dict[Tuple[int, int], List[Tuple[str, bytes, int, int]]] = {}
    vector_pack_entries: Dict[Tuple[int, int], List[Tuple[str, bytes]]] = {}
    tile_count = 0
    vector_tile_count = 0
    vector_feature_count = 0
    max_value = float(np.nanmax(grid.values)) if grid.values.size else 0.0

    for lat_start in tile_range(grid.latitudes, TILE_SIZE):
        lat_end = min(lat_start + TILE_SIZE, 90)
        lat_mask = (grid.latitudes >= lat_start) & (grid.latitudes <= lat_end)
        if not np.any(lat_mask):
            continue
        for lon_start in tile_range(grid.longitudes, TILE_SIZE):
            lon_end = min(lon_start + TILE_SIZE, 180)
            lon_mask = (grid.longitudes >= lon_start) & (grid.longitudes <= lon_end)
            if not np.any(lon_mask):
                continue
            tile_values = grid.values[np.ix_(lat_mask, lon_mask)]
            if tile_values.size == 0 or float(np.nanmax(tile_values)) <= 0:
                continue

            lat_vals = grid.latitudes[lat_mask]
            lon_vals = grid.longitudes[lon_mask]
            dy = (lat_vals[-1] - lat_vals[0]) / (len(lat_vals) - 1) if len(lat_vals) > 1 else 1.0
            dx = (lon_vals[-1] - lon_vals[0]) / (len(lon_vals) - 1) if len(lon_vals) > 1 else 1.0
            tile_bytes = build_binary_tile_bytes(lon_vals, lat_vals, dx, dy, tile_values)
            pack_start = pack_origin(lat_start, lon_start)
            pack_entries.setdefault(pack_start, []).append(
                (tile_key(lat_start, lon_start), tile_bytes, lat_start, lon_start)
            )
            tile_count += 1

            vector_bytes = vector_tile_bytes(lon_vals, lat_vals, dx, dy, tile_values)
            if vector_bytes:
                vector_pack_entries.setdefault(pack_start, []).append((tile_key(lat_start, lon_start), vector_bytes))
                vector_tile_count += 1
                try:
                    vector_feature_count += len(json.loads(vector_bytes.decode("utf-8")).get("features", []))
                except json.JSONDecodeError:
                    pass

    pack_count = 0
    for (pack_lat_start, pack_lon_start), pack_tiles in pack_entries.items():
        entries = [(key, tile_bytes) for key, tile_bytes, _, _ in pack_tiles]
        pack_lat_end = min(pack_lat_start + PACK_LAT_SIZE, 90)
        pack_lon_end = min(pack_lon_start + PACK_LON_SIZE, 180)
        pack_lat_mask = (grid.latitudes >= pack_lat_start) & (grid.latitudes <= pack_lat_end)
        pack_lon_mask = (grid.longitudes >= pack_lon_start) & (grid.longitudes <= pack_lon_end)
        if np.any(pack_lat_mask) and np.any(pack_lon_mask):
            overview_values = grid.values[np.ix_(pack_lat_mask, pack_lon_mask)]
            overview_lat = grid.latitudes[pack_lat_mask]
            overview_lon = grid.longitudes[pack_lon_mask]
            overview_tile = build_overview_tile(overview_lon, overview_lat, overview_values)
            if overview_tile:
                entries.append((overview_tile_key(pack_lat_start, pack_lon_start), overview_tile))
        pack_path = os.path.join(provider_dir, pack_filename(grid.provider_id, 0, pack_lat_start, pack_lon_start))
        write_tile_pack(pack_path, entries)
        pack_count += 1

    vector_pack_count = 0
    for (pack_lat_start, pack_lon_start), pack_tiles in vector_pack_entries.items():
        pack_path = os.path.join(
            vector_provider_dir,
            pack_filename(grid.provider_id, 0, pack_lat_start, pack_lon_start),
        )
        write_tile_pack(pack_path, pack_tiles)
        vector_pack_count += 1

    return {
        "id": grid.provider_id,
        "name": grid.name,
        "status": grid.status,
        "region": grid.region,
        "bounds": REGION_BOUNDS.get(grid.region),
        "ref_time": grid.ref_time,
        "tiles": tile_count,
        "packs": pack_count,
        "max_value": round(max_value, 1),
        "units": RADAR_VALUE_UNITS,
        "value_parameter": RADAR_VALUE_PARAMETER,
        "value_units": RADAR_VALUE_UNITS,
        "source_units": grid.source_units,
        "source_parameter": grid.source_parameter,
        "path": f"radar/{grid.provider_id}",
        "vector_path": f"radar_vectors/{grid.provider_id}",
        "vector_tiles": vector_tile_count,
        "vector_packs": vector_pack_count,
        "vector_features": vector_feature_count,
        "attribution": grid.attribution,
        "source_url": grid.source_url,
        "provider_url": grid.source_url,
        "candidate_urls": candidate_urls(grid.region, grid.source_url),
        **provider_endpoint(grid.region),
        "tile_format": TILE_FORMAT,
        "pack_format": PACK_FORMAT,
        "vector_tile_format": VECTOR_TILE_FORMAT,
        "vector_pack_format": VECTOR_PACK_FORMAT,
        "metadata": {
            "parameter": RADAR_VALUE_PARAMETER,
            "units": RADAR_VALUE_UNITS,
            "value_range": [0, max(1, round(max_value, 1))],
            "target_degrees": grid.target_degrees,
            "client_coloring": True,
            "recommended_color_stops": RADAR_COLOR_STOPS,
        },
    }


def provider_error(provider_id: str, name: str, region: str, source_url: str, attribution: str, exc: Exception) -> Dict[str, Any]:
    return {
        "id": provider_id,
        "name": name,
        "status": "error",
        "region": region,
        "bounds": REGION_BOUNDS.get(region),
        "error": str(exc),
        "tiles": 0,
        "packs": 0,
        "units": RADAR_VALUE_UNITS,
        "value_parameter": RADAR_VALUE_PARAMETER,
        "value_units": RADAR_VALUE_UNITS,
        "path": f"radar/{provider_id}",
        "vector_path": f"radar_vectors/{provider_id}",
        "vector_tiles": 0,
        "vector_packs": 0,
        "vector_features": 0,
        "attribution": attribution,
        "source_url": source_url,
        "provider_url": source_url,
        "candidate_urls": candidate_urls(region, source_url),
        **provider_endpoint(region),
    }


def configuration_pending_manifest(template: Dict[str, Any]) -> Dict[str, Any]:
    endpoint = provider_endpoint(template["region"])
    candidates = candidate_urls(template["region"], template["source_url"])
    return {
        "id": template["id"],
        "name": template["name"],
        "status": "configuration_required",
        "region": template["region"],
        "tiles": 0,
        "packs": 0,
        "units": RADAR_VALUE_UNITS,
        "value_parameter": RADAR_VALUE_PARAMETER,
        "value_units": RADAR_VALUE_UNITS,
        "path": f"radar/{template['id']}",
        "vector_path": f"radar_vectors/{template['id']}",
        "vector_tiles": 0,
        "vector_packs": 0,
        "vector_features": 0,
        "attribution": template["attribution"],
        "source_url": template["source_url"],
        "provider_url": template["source_url"],
        "candidate_urls": candidates,
        **endpoint,
        "bounds": REGION_BOUNDS.get(template["region"]),
        "notes": template["notes"],
    }


def read_configured_provider_specs() -> List[Dict[str, Any]]:
    raw = os.getenv("RADAR_PROVIDERS_JSON")
    if not raw or not raw.strip():
        return []
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("providers", [])
    if not isinstance(payload, list):
        raise ValueError("RADAR_PROVIDERS_JSON must be a JSON list or an object with providers")
    return payload


def layer_entries(provider_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "layers": [
            {
                "id": result["id"],
                "name": result["name"],
                "type": "gusty-grid-pack",
                "region": result["region"],
                "bounds": result.get("bounds"),
                "status": result["status"],
                "path": result["path"],
                "vector_path": result.get("vector_path"),
                "units": RADAR_VALUE_UNITS,
                "value_parameter": result.get("value_parameter", RADAR_VALUE_PARAMETER),
                "value_units": result.get("value_units", RADAR_VALUE_UNITS),
                "source_units": result.get("source_units"),
                "source_parameter": result.get("source_parameter"),
                "tiles": result.get("tiles", 0),
                "packs": result.get("packs", 0),
                "vector_tiles": result.get("vector_tiles", 0),
                "vector_packs": result.get("vector_packs", 0),
                "vector_features": result.get("vector_features", 0),
                "ref_time": result.get("ref_time"),
                "attribution": result.get("attribution"),
                "source_url": result.get("source_url"),
                "provider_url": result.get("provider_url") or result.get("source_url"),
                "candidate_urls": result.get("candidate_urls"),
                "endpoint_url": result.get("endpoint_url"),
                "endpoint_type": result.get("endpoint_type"),
                "machine_readable": result.get("machine_readable"),
                "preferred_raw_product": result.get("preferred_raw_product"),
                "reflectivity_product": result.get("reflectivity_product"),
                "metadata": result.get("metadata"),
                "vector_tile_format": result.get("vector_tile_format"),
                "vector_pack_format": result.get("vector_pack_format"),
            }
            for result in provider_results
        ],
    }


def vector_layer_entries(provider_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "value_parameter": RADAR_VALUE_PARAMETER,
        "value_units": RADAR_VALUE_UNITS,
        "recommended_color_stops": RADAR_COLOR_STOPS,
        "layers": [
            {
                "id": f"{result['id']}_vectors",
                "source_layer_id": result["id"],
                "name": f"{result['name']} Vectors",
                "type": "gusty-vector-pack",
                "region": result["region"],
                "bounds": result.get("bounds"),
                "status": result["status"],
                "path": result.get("vector_path"),
                "value_parameter": result.get("value_parameter", RADAR_VALUE_PARAMETER),
                "value_units": result.get("value_units", RADAR_VALUE_UNITS),
                "tiles": result.get("vector_tiles", 0),
                "packs": result.get("vector_packs", 0),
                "features": result.get("vector_features", 0),
                "ref_time": result.get("ref_time"),
                "attribution": result.get("attribution"),
                "source_url": result.get("source_url"),
                "provider_url": result.get("provider_url") or result.get("source_url"),
                "preferred_raw_product": result.get("preferred_raw_product"),
                "reflectivity_product": result.get("reflectivity_product"),
                "vector_tile_format": result.get("vector_tile_format", VECTOR_TILE_FORMAT),
                "vector_pack_format": result.get("vector_pack_format", VECTOR_PACK_FORMAT),
            }
            for result in provider_results
        ],
    }


def provider_catalog() -> Dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "reference_basis": RADAR_SOURCE_REFERENCE,
        "value_parameter": RADAR_VALUE_PARAMETER,
        "value_units": RADAR_VALUE_UNITS,
        "providers": [
            {
                "id": template["id"],
                "name": template["name"],
                "region": template["region"],
                "bounds": REGION_BOUNDS.get(template["region"]),
                "attribution": template["attribution"],
                "source_url": template["source_url"],
                "provider_url": template["source_url"],
                "candidate_urls": candidate_urls(template["region"], template["source_url"]),
                **provider_endpoint(template["region"]),
                "notes": template["notes"],
            }
            for template in CONFIGURABLE_RADAR_REGIONS
        ],
    }


def run(print_summary: bool = False) -> Dict[str, Any]:
    client = session()
    if os.path.exists(RADAR_DIR):
        shutil.rmtree(RADAR_DIR)
    if os.path.exists(RADAR_VECTOR_DIR):
        shutil.rmtree(RADAR_VECTOR_DIR)
    os.makedirs(RADAR_DIR, exist_ok=True)
    os.makedirs(RADAR_VECTOR_DIR, exist_ok=True)
    os.makedirs(LAYER_DIR, exist_ok=True)

    results: List[Dict[str, Any]] = []
    result_ids = set()
    mrms_available = False
    try:
        result = generate_tiles(fetch_mrms_precip_rate(client))
        results.append(result)
        result_ids.add(result["id"])
        mrms_available = True
    except Exception as exc:
        result = provider_error(
            "radar_north_america_mrms",
            "North America MRMS Rain Rate",
            "North America",
            MRMS_DOCS_URL,
            "NOAA MRMS / AWS Open Data",
            exc,
        )
        results.append(result)
        result_ids.add(result["id"])

    if not mrms_available:
        try:
            result = generate_tiles(fetch_us_iem_nexrad(client))
            results.append(result)
            result_ids.add(result["id"])
        except Exception as exc:
            result = provider_error(
                "radar_us_iem_nexrad",
                "US NEXRAD Rain Rate",
                "United States",
                IEM_N0Q_DOCS_URL,
                "Iowa Environmental Mesonet / NOAA NEXRAD",
                exc,
            )
            results.append(result)
            result_ids.add(result["id"])

    try:
        result = generate_tiles(fetch_canada_geomet_radar(client))
        results.append(result)
        result_ids.add(result["id"])
    except Exception as exc:
        result = provider_error(
            "radar_canada_configured",
            "Canada Radar Intensity",
            "Canada",
            ECCC_GEOMET_URL,
            "Environment and Climate Change Canada / MSC GeoMet",
            exc,
        )
        results.append(result)
        result_ids.add(result["id"])

    try:
        result = generate_tiles(fetch_opera_cirrus_radar(client))
        results.append(result)
        result_ids.add(result["id"])
    except Exception as exc:
        result = provider_error(
            "radar_europe_opera_cirrus",
            "Europe OPERA/CIRRUS Rain Rate",
            "Europe",
            OPERA_CIRRUS_ANIMATOR_URL,
            "EUMETNET OPERA / Finnish Meteorological Institute",
            exc,
        )
        results.append(result)
        result_ids.add(result["id"])

    try:
        result = generate_tiles(fetch_jma_nowcast(client))
        results.append(result)
        result_ids.add(result["id"])
    except Exception as exc:
        result = provider_error(
            "radar_japan_configured",
            "Japan Radar Intensity",
            "Japan",
            "https://www.jma.go.jp/jp/radnowc/index.html",
            "Japan Meteorological Agency",
            exc,
        )
        results.append(result)
        result_ids.add(result["id"])

    try:
        configured_specs = read_configured_provider_specs()
    except Exception as exc:
        configured_specs = []
        result = provider_error(
            "radar_configured_providers",
            "Configured Radar Providers",
            "Configured regions",
            RADAR_SOURCE_REFERENCE,
            "Configured radar providers",
            exc,
        )
        results.append(result)
        result_ids.add(result["id"])

    for spec in configured_specs:
        provider_id = str(spec.get("id", "")).strip()
        if not provider_id:
            continue
        try:
            result = generate_tiles(fetch_configured_radar(client, spec))
        except Exception as exc:
            result = provider_error(
                provider_id,
                spec.get("name", provider_id),
                spec.get("region", "Configured radar"),
                spec.get("source_url", RADAR_SOURCE_REFERENCE),
                spec.get("attribution", provider_id),
                exc,
            )
        results.append(result)
        result_ids.add(result["id"])

    try:
        uk_grid = fetch_configured_uk_radar(client)
        if uk_grid and uk_grid.provider_id not in result_ids:
            result = generate_tiles(uk_grid)
            results.append(result)
            result_ids.add(result["id"])
    except Exception as exc:
        result = provider_error(
            "radar_uk_configured",
            "UK Radar Reflectivity",
            "United Kingdom",
            os.getenv("UK_RADAR_SOURCE_URL", MET_OFFICE_PUBLIC_RADAR_URL),
            os.getenv("UK_RADAR_ATTRIBUTION", "Met Office"),
            exc,
        )
        results.append(result)
        result_ids.add(result["id"])

    for template in CONFIGURABLE_RADAR_REGIONS:
        if template["id"] not in result_ids:
            result = configuration_pending_manifest(template)
            results.append(result)
            result_ids.add(result["id"])

    manifest = {
        "generated_at": now_iso(),
        "layers_file": "layers/radar_layers.json",
        "vector_layers_file": "layers/radar_vector_layers.json",
        "reference_basis": RADAR_SOURCE_REFERENCE,
        "value_parameter": RADAR_VALUE_PARAMETER,
        "value_units": RADAR_VALUE_UNITS,
        "recommended_color_stops": RADAR_COLOR_STOPS,
        "configured_provider_count": len(configured_specs),
        "providers": results,
    }
    write_json(os.path.join(RADAR_DIR, "manifest.json"), manifest)
    write_json(os.path.join(RADAR_DIR, "provider_catalog.json"), provider_catalog())
    write_json(os.path.join(LAYER_DIR, "radar_layers.json"), layer_entries(results))
    write_json(os.path.join(LAYER_DIR, "radar_vector_layers.json"), vector_layer_entries(results))
    if print_summary:
        print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    run(args.print_summary)


if __name__ == "__main__":
    main()
