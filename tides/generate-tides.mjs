import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { allStations, qualityMap, stations } from "@neaps/tide-database";
import { createTidePredictor } from "@neaps/tide-predictor";

const DEFAULT_DAYS = 45;
const NOT_FOR_NAVIGATION =
  "Tide predictions are informational and not for navigation.";

const stationOverrides = {
  "ticon/scarborough-sca-gbr-cco": {
    name: "Scarborough",
  },
};

function parseArgs(argv) {
  const args = {
    days: DEFAULT_DAYS,
    outputDir: path.resolve("public/data/tides"),
    stationIds: new Set(),
    maxStations: undefined,
    printSummary: false,
    includeNonCommercial:
      process.env.TIDES_INCLUDE_NON_COMMERCIAL?.toLowerCase() !== "false",
    includeAllQuality:
      process.env.TIDES_INCLUDE_ALL_QUALITY?.toLowerCase() !== "false",
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--days") {
      args.days = parsePositiveInt(next, "--days");
      index += 1;
    } else if (arg === "--output-dir") {
      if (!next) throw new Error("--output-dir requires a value");
      args.outputDir = path.resolve(next);
      index += 1;
    } else if (arg === "--station-id") {
      if (!next) throw new Error("--station-id requires a value");
      args.stationIds.add(next);
      index += 1;
    } else if (arg === "--max-stations") {
      args.maxStations = parsePositiveInt(next, "--max-stations");
      index += 1;
    } else if (arg === "--print-summary") {
      args.printSummary = true;
    } else if (arg === "--exclude-non-commercial" || arg === "--commercial-only") {
      args.includeNonCommercial = false;
    } else if (arg === "--quality-accepted-only") {
      args.includeAllQuality = false;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return args;
}

function parsePositiveInt(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function startOfTodayUtc(now = new Date()) {
  return new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), 0, 0, 0),
  );
}

function addDays(date, days) {
  return new Date(date.getTime() + days * 24 * 60 * 60 * 1000);
}

function round(value, places = 3) {
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
}

function stationName(station) {
  return stationOverrides[station.id]?.name ?? station.name;
}

function chartDatumOffset(station) {
  const datums = station.datums ?? {};
  const msl = datums.MSL ?? 0;
  const chartDatum = station.chart_datum ? datums[station.chart_datum] : undefined;
  if (typeof chartDatum !== "number") return 0;
  return msl - chartDatum;
}

function isReferenceWithHarmonics(station) {
  return (
    station.type === "reference" &&
    Array.isArray(station.harmonic_constituents) &&
    station.harmonic_constituents.length > 0
  );
}

function isAllowedForCurrentMode(station, args) {
  if (!isReferenceWithHarmonics(station)) return false;
  if (!args.includeNonCommercial && station.license?.commercial_use !== true) {
    return false;
  }
  return true;
}

function compactStation(station) {
  return {
    id: station.id,
    name: stationName(station),
    sourceName: station.source?.name ?? null,
    sourceId: station.source?.id ?? null,
    sourceUrl: station.source?.url ?? null,
    license: station.license?.type ?? null,
    licenseUrl: station.license?.url ?? null,
    commercialUse: station.license?.commercial_use === true,
    country: station.country,
    region: station.region ?? null,
    continent: station.continent ?? null,
    timezone: station.timezone ?? "UTC",
    latitude: station.latitude,
    longitude: station.longitude,
    chartDatum: station.chart_datum ?? null,
    datumsSource: station.datums_source ?? null,
    qualityAccepted: qualityMap.get(station.id)?.accepted ?? false,
    quality: station.disclaimers || "No obvious issues",
  };
}

function eventFileName(stationId) {
  return `${stationId.replaceAll("/", "__")}.json`;
}

function buildEvents(station, start, end) {
  const predictor = createTidePredictor(station.harmonic_constituents, {
    offset: chartDatumOffset(station),
  });
  return predictor
    .getExtremesPrediction({ start, end })
    .map((event) => ({
      time: event.time.toISOString(),
      type: event.high ? "high" : "low",
      heightM: round(event.level, 3),
    }))
    .sort((a, b) => a.time.localeCompare(b.time));
}

async function writeJson(filePath, data) {
  await writeFile(`${filePath}.tmp`, `${JSON.stringify(data, null, 2)}\n`);
  await rm(filePath, { force: true });
  await writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`);
  await rm(`${filePath}.tmp`, { force: true });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const generatedAt = new Date();
  const validFrom = startOfTodayUtc(generatedAt);
  const validTo = addDays(validFrom, args.days);

  const stationSource = args.includeAllQuality ? allStations : stations;
  let selected = stationSource.filter((station) =>
    isAllowedForCurrentMode(station, args),
  );
  if (args.stationIds.size > 0) {
    selected = selected.filter((station) => args.stationIds.has(station.id));
  }
  selected.sort((a, b) => a.id.localeCompare(b.id));
  if (args.maxStations !== undefined) {
    selected = selected.slice(0, args.maxStations);
  }

  await rm(args.outputDir, { recursive: true, force: true });
  await mkdir(path.join(args.outputDir, "events"), { recursive: true });

  const stationIndex = [];
  const errors = [];

  for (const station of selected) {
    try {
      const compact = compactStation(station);
      const events = buildEvents(station, validFrom, validTo);
      const fileName = eventFileName(station.id);
      stationIndex.push({
        ...compact,
        eventPath: `events/${fileName}`,
        eventCount: events.length,
      });
      await writeJson(path.join(args.outputDir, "events", fileName), {
        station: compact,
        generatedAt: generatedAt.toISOString(),
        validFrom: validFrom.toISOString(),
        validTo: validTo.toISOString(),
        days: args.days,
        units: {
          height: "m above chart datum",
          time: "UTC",
        },
        disclaimer: NOT_FOR_NAVIGATION,
        attribution:
          "Tide harmonic constituents from the Neaps tide database (https://github.com/openwatersio/tide-database).",
        events,
      });
    } catch (error) {
      errors.push({
        id: station.id,
        name: station.name,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  await writeJson(path.join(args.outputDir, "stations.json"), stationIndex);
  await writeJson(path.join(args.outputDir, "manifest.json"), {
    generatedAt: generatedAt.toISOString(),
    validFrom: validFrom.toISOString(),
    validTo: validTo.toISOString(),
    days: args.days,
    stationCount: stationIndex.length,
    errorCount: errors.length,
    files: {
      stations: "stations.json",
      events: "events/<station-id-with-slashes-replaced-by-__.json>",
    },
    filters: {
      commercialUse: args.includeNonCommercial ? "included" : true,
      qualityAcceptedOnly: !args.includeAllQuality,
      stationType: "reference",
      requiresHarmonicConstituents: true,
    },
    units: {
      height: "m above chart datum",
      time: "UTC",
    },
    disclaimer: NOT_FOR_NAVIGATION,
    attribution:
      "Tide harmonic constituents from the Neaps tide database (https://github.com/openwatersio/tide-database).",
    errors,
  });

  const summary = {
    outputDir: args.outputDir,
    stationCount: stationIndex.length,
    errorCount: errors.length,
    includeNonCommercial: args.includeNonCommercial,
    includeAllQuality: args.includeAllQuality,
    generatedAt: generatedAt.toISOString(),
    validFrom: validFrom.toISOString(),
    validTo: validTo.toISOString(),
  };

  if (args.printSummary) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    console.log(
      `Generated ${summary.stationCount} tide station files in ${args.outputDir}`,
    );
  }

  if (errors.length > 0) {
    console.error(`Skipped ${errors.length} stations with prediction errors.`);
  }
}

const entryPath = fileURLToPath(import.meta.url);
if (process.argv[1] === entryPath) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
