#!/usr/bin/env python3
"""Stream a directory to a versioned R2 prefix, then atomically promote it."""

import argparse
import concurrent.futures
import datetime as dt
import gzip
import json
import mimetypes
import os
import pathlib
import tempfile

import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig


def client(account_id, access_key, secret_key):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            connect_timeout=10,
            read_timeout=120,
            max_pool_connections=32,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def upload_file(s3, bucket, source, key, compresslevel):
    content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    with tempfile.NamedTemporaryFile(suffix=".gz") as compressed:
        with open(source, "rb") as input_file, gzip.GzipFile(
            fileobj=compressed, mode="wb", compresslevel=compresslevel, mtime=0
        ) as output_file:
            while chunk := input_file.read(1024 * 1024):
                output_file.write(chunk)
        compressed.flush()
        s3.upload_file(
            compressed.name,
            bucket,
            key,
            ExtraArgs={
                "ContentEncoding": "gzip",
                "ContentType": content_type,
                "CacheControl": "public,max-age=31536000,immutable",
            },
            Config=TransferConfig(max_concurrency=1),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compression-level", type=int, default=1)
    args = parser.parse_args()
    root = pathlib.Path(args.source)
    files = [path for path in root.rglob("*") if path.is_file()]
    prefix = f"{args.dataset}/runs/{args.run}"
    s3 = client(
        os.environ["CLOUDFLARE_ACCOUNT_ID"],
        os.environ["R2_ACCESS_KEY_ID"],
        os.environ["R2_SECRET_ACCESS_KEY"],
    )

    def upload(path):
        upload_file(
            s3,
            args.bucket,
            path,
            f"{prefix}/{path.relative_to(root)}",
            args.compression_level,
        )

    print(json.dumps({
        "prefix": prefix,
        "files": len(files),
        "workers": args.workers,
        "compression_level": args.compression_level,
    }), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(upload, path) for path in files]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            future.result()
            if completed % 50 == 0 or completed == len(files):
                print(f"Uploaded {completed}/{len(files)} files", flush=True)

    pointer = json.dumps({
        "schema": 1,
        "run": f"runs/{args.run}",
        "published_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": len(files),
    }, separators=(",", ":")).encode()
    s3.put_object(
        Bucket=args.bucket,
        Key=f"{args.dataset}/current.json",
        Body=pointer,
        ContentType="application/json",
        CacheControl="no-cache, max-age=0",
    )
    print(json.dumps({"prefix": prefix, "files": len(files)}))


if __name__ == "__main__":
    main()
