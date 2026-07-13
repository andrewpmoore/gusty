"""Shared timestamp parsing for weather-provider and observation data."""

import datetime as dt
import re


def parse_utc(value):
    text = str(value).strip()
    # NumPy/xarray timestamps commonly contain nanoseconds; datetime supports 6 digits.
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = dt.datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
