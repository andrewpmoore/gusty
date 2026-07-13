#!/usr/bin/env python3
"""Mirror Gusty's public scene bundle into an R2-ready directory."""

import argparse
import json
import pathlib
import re
import urllib.request


SCENE_ID = re.compile(r"^[a-z0-9_-]+$")


def download(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "Gusty Weather scene sync/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def sync(source_base, public_base, output):
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(download(f"{source_base}/manifest.json"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Scene manifest is empty or invalid")

    rewritten = []
    for item in manifest:
        scene_id = str(item.get("id", ""))
        if not SCENE_ID.fullmatch(scene_id):
            raise ValueError(f"Invalid scene id: {scene_id!r}")
        scene = download(f"{source_base}/{scene_id}.json")
        json.loads(scene)
        image = download(f"{source_base}/{scene_id}.webp")
        if not image.startswith(b"RIFF") or image[8:12] != b"WEBP":
            raise ValueError(f"Invalid WebP image for {scene_id}")
        (output / f"{scene_id}.json").write_bytes(scene)
        (output / f"{scene_id}.webp").write_bytes(image)
        rewritten.append({
            **item,
            "thumbnailUrl": f"{public_base}/{scene_id}.webp",
        })
    (output / "manifest.json").write_text(
        json.dumps(rewritten, separators=(",", ":")) + "\n"
    )
    return len(rewritten)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-base", required=True)
    parser.add_argument("--public-base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = sync(
        args.source_base.rstrip("/"),
        args.public_base.rstrip("/"),
        args.output,
    )
    print(json.dumps({"scenes": count, "output": args.output}))


if __name__ == "__main__":
    main()
