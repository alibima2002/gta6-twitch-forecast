// gta6-twitch-ccu-forecast-01: launch-day Twitch-viewer forecaster.
//
// Idea: a launch day on Twitch follows the unlock clock. A game that unlocks at
// midnight local time, country by country, opens its UTC day with only the
// timezones already past midnight watching, dips in the small UTC hours, then
// climbs as the rest of the world wakes and peaks in the US evening. So the
// day's shape, as a ratio to the first hour's print, is predictable from past
// launches that unlocked the same way — but not from launches that unlocked all
// at once (their first hour already holds everyone) or from launches whose
// early access had players in days before release (their first hour is
// inflated). This program therefore learns the shape only from past
// rolling-midnight launches, and conditions the shape on how big the first
// hour was: a small first hour means most of the audience is still to arrive
// and the day climbs steeply; a huge first hour means the audience is already
// there and the day stays flat. Two views of that conditioning are blended: a
// log-log regression of each hour's log ratio on the log first hour, and the
// median log ratio of the eight most similar first hours. The spread comes
// from the disagreement among those eight neighbours, widened a little, so the
// quantiles say honestly how uncertain the day is.
//
// Everything the forecast needs is inside the case file: the first hour's
// print and the comparables' hourly histories. No network, no clock, no
// randomness: the same case always gives the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** One hour, in milliseconds. */
const HOUR_MS = 3600_000;
/** How many nearest first-hours the local view uses. */
const NEAREST = 8;
/** Weight on the local (nearest-neighbour) median; the rest is the regression. */
const LOCAL_WEIGHT = 0.3;
/** How much the neighbours' spread is widened before it becomes the quantiles. */
const SPREAD_WIDEN = 1.25;
/** The narrowest spread allowed, in natural-log units. */
const MIN_SPREAD = 0.12;
/** The house baseline's spread, used only when no donor has the hour. */
const FALLBACK_SIGMA = 0.15;

interface HourlyPoint {
  from: string;
  viewers: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  id: string;
  game: { name: string; twitch_id: string };
  day_start: string;
  checkpoint: string;
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  history_hourly?: Array<{ from: string; viewers: number }>;
  comparables?: Comparable[];
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;
/** The middle value of a sorted list (lower median for an even count). */
const median = (sorted: number[]): number => sorted[Math.floor(sorted.length / 2)];
/** The p-quantile of a sorted list, by nearest rank. */
const quantile = (sorted: number[], p: number): number =>
  sorted[Math.max(0, Math.min(sorted.length - 1, Math.floor(p * sorted.length)))];
/** The numbers of a list, sorted ascending. */
const sorted = (xs: number[]): number[] => [...xs].sort((a, b) => a - b);

/**
 * A launch's release-day curve: the mean viewers of each of the 24 UTC hours
 * after day_start, or null for an hour with no point. The first entry is the
 * day's first hour, the anchor every later hour is ratioed against.
 */
const dayCurve = (comp: Comparable): Array<number | null> => {
  const start = Date.parse(comp.day_start);
  if (Number.isNaN(start)) return Array(24).fill(null);
  const sums: number[] = Array(24).fill(0);
  const counts: number[] = Array(24).fill(0);
  for (const p of comp.hourly ?? []) {
    if (p.viewers == null || p.viewers <= 0) continue;
    const h = Math.round((Date.parse(p.from) - start) / HOUR_MS) + 1;
    if (h < 1 || h > 24) continue;
    sums[h - 1] += p.viewers;
    counts[h - 1] += 1;
  }
  return sums.map((s, i) => (counts[i] > 0 ? s / counts[i] : null));
};

/**
 * Is this comparable a rolling-midnight launch like the target? True when its
 * unlock note says midnight local / rolling and it is neither early-access
 * contaminated, nor a weak analogue, nor a simultaneous global unlock. The
 * target's own pre-release category is never a donor: it has no release day.
 */
const isRollingLaunch = (comp: Comparable, targetName: string): boolean => {
  if (comp.name.toLowerCase().includes(targetName.toLowerCase())) return false;
  const u = (comp.unlock ?? '').toLowerCase();
  if (/early-access-contaminated/.test(u)) return false;
  if (/tier:\s*weak/.test(u)) return false;
  if (
    /single global unlock|single simultaneous global|simultaneous across platforms|simultaneous global|simultaneous midnight gmt|standard edition simultaneous global/.test(
      u,
    )
  ) {
    return false;
  }
  return /rolling|midnight local/.test(u);
};

/** A donor: a past rolling-midnight launch with its release-day curve. */
interface Donor {
  firstHour: number;
  curve: Array<number | null>;
}

/** The persistence row the house baseline would write for one hour. */
const baselineRow = (p1: number, h: number): number[] =>
  Z.map((z) => round2(p1 * Math.exp(z * FALLBACK_SIGMA * Math.sqrt(h - 1))));

/**
 * The five quantiles for one hour: the day's shape at that hour, as a ratio
 * to the first hour's print, learned from the donors. The log ratio is
 * predicted two ways and blended: a log-log regression on the first hour's
 * size, and the median of the nearest first hours. The spread is the
 * neighbours' interquartile range, widened, so the band is honest about how
 * much launch days differ even among similar first hours.
 */
const hourQuantiles = (p1: number, h: number, donors: Donor[]): number[] => {
  const points = donors
    .filter((d) => d.curve[h - 1] != null && d.curve[0] != null && d.curve[0]! > 0)
    .map((d) => ({ x: Math.log(d.firstHour), y: Math.log(d.curve[h - 1]! / d.curve[0]!) }));
  if (points.length === 0) return baselineRow(p1, h);
  const n = points.length;
  const meanX = points.reduce((s, p) => s + p.x, 0) / n;
  const meanY = points.reduce((s, p) => s + p.y, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (const p of points) {
    sxy += (p.x - meanX) * (p.y - meanY);
    sxx += (p.x - meanX) * (p.x - meanX);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  const intercept = meanY - slope * meanX;
  const x0 = Math.log(p1);
  const regressed = intercept + slope * x0;
  const neighbours = [...points]
    .sort((a, b) => Math.abs(a.x - x0) - Math.abs(b.x - x0))
    .slice(0, Math.min(NEAREST, points.length));
  const nearLogs = sorted(neighbours.map((p) => p.y));
  const local = median(nearLogs);
  const spread = Math.max(
    MIN_SPREAD,
    (quantile(nearLogs, 0.75) - quantile(nearLogs, 0.25)) / 1.349,
  );
  const centre = (1 - LOCAL_WEIGHT) * regressed + LOCAL_WEIGHT * local;
  const qs = Z.map((z) => round2(p1 * Math.exp(centre + z * spread * SPREAD_WIDEN)));
  // Keep the row finite, positive, bounded and non-decreasing, whatever the data did.
  let prev = 1;
  return qs.map((q) => {
    const v = Number.isFinite(q) ? Math.min(1_000_000_000, Math.max(1, q)) : prev;
    prev = Math.max(prev, v);
    return prev;
  });
};

/** The whole forecast for one case: 23 rows, hours 2 to 24. */
const forecastCase = (c: CaseFile): { hours: Array<{ hour: number; q: number[] }> } => {
  const first = c.known.find((k) => k.hour === 1);
  const lastHistory = (c.history_hourly ?? []).at(-1);
  const p1 = Math.max(1, first?.viewers ?? lastHistory?.viewers ?? 1);
  const rolling: Donor[] = [];
  const all: Donor[] = [];
  for (const comp of c.comparables ?? []) {
    const curve = dayCurve(comp);
    if (curve[0] == null || curve[0] <= 0) continue;
    const donor: Donor = { firstHour: curve[0], curve };
    all.push(donor);
    if (isRollingLaunch(comp, c.game.name)) rolling.push(donor);
  }
  // Prefer the rolling-midnight donors; fall back to every launch if none qualify.
  const donors = rolling.length > 0 ? rolling : all;
  const hours = c.forecast.map(({ hour }) => ({ hour, q: hourQuantiles(p1, hour, donors) }));
  return { hours };
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(forecastCase(c))}\n`);
}
