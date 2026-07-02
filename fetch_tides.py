import argparse
import os
import subprocess
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TIDES_DIR = os.path.join(ROOT_DIR, "tides")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate static tide prediction JSON for Gusty."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=45,
        help="Number of days of tide events to generate from today UTC.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("public", "data", "tides"),
        help="Output directory for generated tide JSON, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--station-id",
        action="append",
        default=[],
        help="Limit generation to a station id. May be passed multiple times.",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Limit station count for local smoke tests.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the generator summary JSON.",
    )
    parser.add_argument(
        "--exclude-non-commercial",
        action="store_true",
        help="Only include stations marked as commercially usable.",
    )
    parser.add_argument(
        "--quality-accepted-only",
        action="store_true",
        help="Only include stations accepted by the Neaps quality filter.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(ROOT_DIR, output_dir)

    command = [
        "node",
        os.path.join(TIDES_DIR, "generate-tides.mjs"),
        "--days",
        str(args.days),
        "--output-dir",
        output_dir,
    ]
    for station_id in args.station_id:
        command.extend(["--station-id", station_id])
    if args.max_stations is not None:
        command.extend(["--max-stations", str(args.max_stations)])
    if args.print_summary:
        command.append("--print-summary")
    if args.exclude_non_commercial:
        command.append("--exclude-non-commercial")
    if args.quality_accepted_only:
        command.append("--quality-accepted-only")

    try:
        subprocess.run(command, cwd=ROOT_DIR, check=True)
    except FileNotFoundError as exc:
        print(f"Missing executable: {exc}", file=sys.stderr)
        return 127
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
