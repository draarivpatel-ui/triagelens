/* TriageLens ML engine. Pure ES module, zero dependencies, fully deterministic.
 *
 * Runs unchanged in the browser and in Node. Contains three synthetic social
 * impact benchmark scenarios, two models trained from scratch (L2 regularized
 * logistic regression and a depth limited Gini decision tree), Platt / temperature
 * calibration fitted on the TRAIN split only, post-processing fairness mitigations,
 * and held out evaluation with a subgroup audit.
 *
 * Every record is synthetic. No row is a real patient, student, or shelter guest.
 */

/* ------------------------------------------------------------------ */
/* Deterministic random numbers and helpers                            */
/* ------------------------------------------------------------------ */

/** mulberry32: 32 bit state, deterministic, uniform in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalSampler(rand) {
  let spare = null;
  return function normal(mean = 0, sd = 1) {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return mean + sd * v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    const th = 2 * Math.PI * v;
    spare = r * Math.sin(th);
    return mean + sd * r * Math.cos(th);
  };
}

function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

function clamp(x, lo, hi) {
  if (!Number.isFinite(x)) return lo;
  return x < lo ? lo : x > hi ? hi : x;
}

export function safeDiv(a, b, fallback = 0) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return fallback;
  const v = a / b;
  return Number.isFinite(v) ? v : fallback;
}

export function sanitizeNumber(x, fallback = 0) {
  if (x === null || x === undefined) return fallback;
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
}

function mean(xs) {
  if (!xs || !xs.length) return 0;
  let s = 0;
  for (const x of xs) s += sanitizeNumber(x, 0);
  return s / xs.length;
}

function stddev(xs) {
  if (!xs || !xs.length) return 1;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (sanitizeNumber(x, 0) - m) ** 2;
  return Math.sqrt(s / Math.max(1, xs.length)) || 1;
}

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

/* ------------------------------------------------------------------ */
/* Synthetic benchmark scenarios                                       */
/* ------------------------------------------------------------------ */

export const SCENARIOS = [
  {
    id: 'clinic',
    name: 'Community Clinic Triage Priority',
    short: 'Clinic',
    positive: 'priority triage',
    outcomeNote:
      'Synthetic eligibility label for priority triage. Rural and low access patients have fewer recorded signs even when need is the same, so historical labels under triage them.',
    features: [
      { key: 'symptomSeverity', label: 'Symptom severity (0 to 100)' },
      { key: 'waitDays', label: 'Days since symptom onset' },
      { key: 'hasProvider', label: 'Usual source of care (0 or 1)' },
      { key: 'ageYears', label: 'Age (years)' },
      { key: 'incomeIndex', label: 'Household income index (0 to 100)' }
    ],
    proxyKey: 'incomeIndex',
    subgroupKey: 'rural',
    subgroupLabel: 'Rural or low access',
    groupLabels: { 0: 'Urban or usual access', 1: 'Rural or low access' },
    harmedGroup: 1,
    ethics: {
      benefits: 'Faster review of urgent synthetic cases and a visible check on rural under triage.',
      harms: 'A rural patient could be ranked too low when sparse records are mistaken for low need.',
      failures: 'Missing history, shifted prevalence, and proxies such as income or transport access.',
      human: 'A clinician must confirm priority. The score is a queue aid, never a diagnosis.'
    }
  },
  {
    id: 'scholarship',
    name: 'Student Scholarship and Tutoring Support',
    short: 'Scholarship',
    positive: 'tutoring support',
    outcomeNote:
      'Synthetic label for tutoring support. Zip code and household income leak into historical awards, creating a proxy for low income students.',
    features: [
      { key: 'gpa', label: 'Grade average (0 to 100)' },
      { key: 'attendance', label: 'Attendance rate (0 to 100)' },
      { key: 'tutoringNeed', label: 'Self reported tutoring need (0 to 100)' },
      { key: 'householdIncome', label: 'Household income index (0 to 100)' },
      { key: 'zipAdvantage', label: 'Zip code advantage index (0 to 100)' }
    ],
    proxyKey: 'zipAdvantage',
    subgroupKey: 'lowIncome',
    subgroupLabel: 'Low income household',
    groupLabels: { 0: 'Higher income household', 1: 'Low income household' },
    harmedGroup: 1,
    ethics: {
      benefits: 'Tutoring capacity can reach students whose need is real but under recorded.',
      harms: 'A low income student can be passed over when zip code stands in for opportunity.',
      failures: 'Proxy features, self reporting gaps, and labels copied from past unequal awards.',
      human: 'A counselor reviews every flag. The score cannot decide who deserves support.'
    }
  },
  {
    id: 'shelter',
    name: 'Emergency Shelter Bed Routing',
    short: 'Shelter',
    positive: 'urgent bed routing',
    outcomeNote:
      'Synthetic label for urgent bed routing. Language barriers suppress reported urgency, so the recorded label understates need for some guests.',
    features: [
      { key: 'exposureHours', label: 'Outdoor exposure (hours)' },
      { key: 'healthRisks', label: 'Health risk count (0 to 8)' },
      { key: 'dependents', label: 'Dependents needing shelter (0 to 5)' },
      { key: 'transportAccess', label: 'Transport access (0 or 1)' },
      { key: 'languageBarrier', label: 'Language barrier flag (0 or 1)' }
    ],
    proxyKey: 'languageBarrier',
    subgroupKey: 'languageBarrier',
    subgroupLabel: 'Language barrier',
    groupLabels: { 0: 'No language barrier', 1: 'Language barrier' },
    harmedGroup: 1,
    ethics: {
      benefits: 'Urgent synthetic placements surface earlier even when a report is incomplete.',
      harms: 'A guest with a language barrier can be routed late when low reported urgency is taken at face value.',
      failures: 'Reporting bias, missing interpreter context, and time pressure in intake.',
      human: 'A trained intake worker confirms urgency. Nobody is denied a bed by a model.'
    }
  }
];

export const DEFAULT_BIAS = 0.6;
export const DEFAULT_SEED = 20270101;
export const DEFAULT_TRAIN_FRACTION = 0.7;
export const DEFAULT_N = 1400;
export const DEFAULT_THRESHOLD = 0.35;

function scenarioById(id) {
  return SCENARIOS.find((s) => s.id === id) || SCENARIOS[0];
}

/** Correlated synthetic features. Each row is generated, not observed. */
function drawRow(rand, normal, scenario) {
  if (scenario.id === 'clinic') {
    const rural = rand() < 0.38 ? 1 : 0;
    const ageYears = clamp(normal(50, 21), 18, 92);
    const incomeIndex = clamp(72 - 30 * rural - 0.35 * (ageYears - 50) + normal(0, 12), 4, 99);
    const hasProvider = rand() < 0.93 - 0.52 * rural ? 1 : 0;
    const symptomSeverity = clamp(
      50 + 0.35 * (ageYears - 50) - 0.22 * (incomeIndex - 55) - 8 * (1 - hasProvider) + normal(0, 16),
      2, 99
    );
    const waitDays = clamp(7 + 0.07 * (ageYears - 50) + 0.18 * (80 - incomeIndex) + normal(0, 5), 0, 48);
    return { rural, ageYears, incomeIndex, hasProvider, symptomSeverity, waitDays };
  }
  if (scenario.id === 'scholarship') {
    const lowIncome = rand() < 0.42 ? 1 : 0;
    const zipAdvantage = clamp(55 + 30 * (1 - lowIncome) + normal(0, 13), 2, 99);
    const householdIncome = clamp(84 - 46 * lowIncome + normal(0, 11), 3, 99);
    const gpa = clamp(72 + 0.26 * (zipAdvantage - 50) - 0.10 * (55 - householdIncome) + normal(0, 9), 25, 100);
    const attendance = clamp(83 + 0.18 * (zipAdvantage - 50) + normal(0, 8), 30, 100);
    const tutoringNeed = clamp(52 - 0.20 * (gpa - 70) + normal(0, 16), 1, 99);
    return { lowIncome, zipAdvantage, householdIncome, gpa, attendance, tutoringNeed };
  }
  const languageBarrier = rand() < 0.34 ? 1 : 0;
  const transportAccess = rand() < 0.81 - 0.18 * languageBarrier ? 1 : 0;
  const exposureHours = clamp(10 + 7 * (1 - transportAccess) + normal(0, 6), 0, 48);
  const healthRisks = clamp(Math.round(2.4 + 0.45 * languageBarrier + normal(0, 1.5)), 0, 8);
  const dependents = clamp(Math.round(1.1 + normal(0, 1.3)), 0, 5);
  return { languageBarrier, transportAccess, exposureHours, healthRisks, dependents };
}

/** Latent priority before bias is applied to the recorded label. */
function latentProbability(row, scenario) {
  if (scenario.id === 'clinic') {
    return sigmoid(
      -0.55 +
        0.050 * (row.symptomSeverity - 50) +
        0.028 * (row.waitDays - 10) +
        0.60 * (1 - row.hasProvider) +
        0.011 * (row.ageYears - 50)
    );
  }
  if (scenario.id === 'scholarship') {
    return sigmoid(
      -0.70 +
        0.050 * (row.gpa - 70) +
        0.050 * (row.tutoringNeed - 55) +
        0.028 * (row.attendance - 82) -
        0.018 * (row.householdIncome - 60)
    );
  }
  return sigmoid(
    -0.90 +
      0.080 * (row.exposureHours - 14) +
      0.34 * row.healthRisks +
      0.22 * row.dependents +
      0.50 * (1 - row.transportAccess)
  );
}

/** bias > 0 subtracts a group penalty from the recorded label probability. */
function labelBias(row, scenario, bias) {
  const b = clamp(bias, 0, 1);
  if (scenario.id === 'clinic') return 0.62 * b * row.rural;
  if (scenario.id === 'scholarship') return 0.60 * b * row.lowIncome;
  return 0.58 * b * row.languageBarrier;
}

/** Stratified deterministic split. Returns copies so callers may annotate rows. */
export function generateDataset({
  scenarioId = 'clinic',
  seed = DEFAULT_SEED,
  bias = DEFAULT_BIAS,
  n = DEFAULT_N,
  trainFraction = DEFAULT_TRAIN_FRACTION
} = {}) {
  const scenario = scenarioById(scenarioId);
  const rand = mulberry32((Number(seed) >>> 0) ^ 0x9e3779b9);
  const normal = normalSampler(rand);
  const count = Math.max(20, Math.floor(n));
  const rows = new Array(count);
  for (let i = 0; i < count; i++) rows[i] = drawRow(rand, normal, scenario);
  for (const row of rows) {
    row.y = latentProbability(row, scenario) - labelBias(row, scenario, bias) > rand() ? 1 : 0;
  }

  const positives = rows.filter((r) => r.y === 1);
  const negatives = rows.filter((r) => r.y === 0);
  const shuffle = (arr) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  };
  shuffle(positives);
  shuffle(negatives);
  const nTrainPos = Math.round(positives.length * trainFraction);
  const nTrainNeg = Math.round(negatives.length * trainFraction);
  const train = positives.slice(0, nTrainPos).concat(negatives.slice(0, nTrainNeg));
  const test = positives.slice(nTrainPos).concat(negatives.slice(nTrainNeg));
  shuffle(train);
  shuffle(test);
  train.forEach((r, i) => { r.id = i + 1; });
  test.forEach((r, i) => { r.id = i + 1; });

  return {
    scenario: scenario.id,
    seed: Number(seed) >>> 0,
    bias: Number(bias),
    n: count,
    trainFraction,
    features: scenario.features.map((f) => ({ key: f.key, label: f.label })),
    subgroupKey: scenario.subgroupKey,
    groupLabels: { ...scenario.groupLabels },
    train,
    test
  };
}

/* ------------------------------------------------------------------ */
/* Logistic regression                                                 */
/* ------------------------------------------------------------------ */

function standardize(rows, keys) {
  const rr = Array.isArray(rows) ? rows : [];
  const stats = keys.map((key) => {
    const xs = rr.map((r) => (r && r[key] != null ? r[key] : 0));
    return { key, mean: mean(xs), sd: stddev(xs) || 1 };
  });
  const z = rr.map((r) => keys.map((key, j) => (sanitizeNumber(r && r[key], 0) - stats[j].mean) / (stats[j].sd || 1)));
  return { stats, z };
}

function standardizeOne(row, stats) {
  // Missing or non-numeric features are mean-imputed (z = 0) so a partial row
  // never produces a NaN probability.
  return stats.map((s) => (sanitizeNumber(row && row[s.key], s.mean) - s.mean) / s.sd);
}

/** Full batch gradient descent on L2 regularized log loss. */
export function trainLogisticRegression(rows, options = {}) {
  const started = nowMs();
  const rr = Array.isArray(rows) ? rows : [];
  const keys = (options && options.featureKeys) || ['symptomSeverity', 'waitDays', 'hasProvider', 'ageYears'];
  const l2 = options.l2 ?? 0.02;
  const epochs = options.epochs ?? 120;
  const lr = options.lr ?? 0.35;
  const { stats, z } = standardize(rr, keys);
  const p = keys.length;
  const w = new Array(p).fill(0);
  let b = 0;
  const loss = [];
  const n = rr.length;
  if (n === 0) {
    const coefficients = {};
    for (const k of keys) coefficients[k] = { raw: 0, standardized: 0, mean: 0, sd: 1 };
    return { kind: 'logistic', keys: keys.slice(), l2, epochs, loss: [], coefficients, intercept: 0, predictProba: () => 0.5, ms: nowMs() - started };
  }
  for (let e = 0; e < epochs; e++) {
    const gw = new Array(p).fill(0);
    let gb = 0;
    let logLoss = 0;
    for (let i = 0; i < n; i++) {
      const xi = z[i];
      const yi = rr[i].y ? 1 : 0;
      let dot = b;
      for (let j = 0; j < p; j++) dot += w[j] * xi[j];
      const q = sigmoid(dot);
      const d = q - yi;
      gb += d;
      for (let j = 0; j < p; j++) gw[j] += d * xi[j];
      const pc = clamp(q, 1e-12, 1 - 1e-12);
      logLoss += -(yi * Math.log(pc) + (1 - yi) * Math.log(1 - pc));
    }
    b -= (lr * gb) / n;
    for (let j = 0; j < p; j++) w[j] -= lr * (gw[j] / n + l2 * w[j]);
    if (e % 10 === 0 || e === epochs - 1) loss.push(logLoss / n);
  }
  const coefficients = {};
  let rawIntercept = b;
  for (let j = 0; j < p; j++) {
    const raw = w[j] / stats[j].sd;
    rawIntercept -= raw * stats[j].mean;
    coefficients[keys[j]] = { raw, standardized: w[j], mean: stats[j].mean, sd: stats[j].sd };
  }
  return {
    kind: 'logistic',
    keys: keys.slice(),
    l2,
    epochs,
    loss,
    coefficients,
    intercept: rawIntercept,
    predictProba(row) {
      const x = standardizeOne(row, stats);
      let dot = b;
      for (let j = 0; j < p; j++) dot += w[j] * x[j];
      return sigmoid(dot);
    },
    ms: nowMs() - started
  };
}

/* ------------------------------------------------------------------ */
/* Decision tree                                                       */
/* ------------------------------------------------------------------ */

function gini(pos, total) {
  if (total <= 0) return 0;
  const q = pos / total;
  return 1 - (q * q + (1 - q) * (1 - q));
}

function bestSplit(rows, keys, minLeaf) {
  let best = null;
  const total = rows.length;
  const pos = rows.reduce((s, r) => s + r.y, 0);
  const parent = gini(pos, total);
  for (const key of keys) {
    const values = rows.map((r) => r[key]);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    if (!(hi > lo)) continue;
    for (let q = 1; q < 12; q++) {
      const t = lo + ((hi - lo) * q) / 12;
      let lp = 0;
      let ln = 0;
      for (const r of rows) {
        if (r[key] <= t) (r.y ? lp++ : ln++);
      }
      const nL = lp + ln;
      const nR = total - nL;
      if (nL < minLeaf || nR < minLeaf) continue;
      const gain =
        parent - (nL / total) * gini(lp, nL) - (nR / total) * gini(pos - lp, nR);
      if (gain > 1e-9 && (!best || gain > best.gain)) best = { key, threshold: t, gain };
    }
  }
  return best;
}

function buildTree(rows, keys, depth, maxDepth, minLeaf) {
  const n = rows.length;
  const pos = rows.reduce((s, r) => s + r.y, 0);
  if (depth >= maxDepth || n < 2 * minLeaf || pos === 0 || pos === n) {
    return { leaf: true, n, positives: pos, p: pos / n };
  }
  const split = bestSplit(rows, keys, minLeaf);
  if (!split) return { leaf: true, n, positives: pos, p: pos / n };
  const left = [];
  const right = [];
  for (const r of rows) (r[split.key] <= split.threshold ? left : right).push(r);
  return {
    leaf: false,
    key: split.key,
    threshold: split.threshold,
    gain: split.gain,
    n,
    positives: pos,
    left: buildTree(left, keys, depth + 1, maxDepth, minLeaf),
    right: buildTree(right, keys, depth + 1, maxDepth, minLeaf)
  };
}

function treeProba(node, row) {
  let cur = node;
  while (!cur.leaf) cur = row[cur.key] <= cur.threshold ? cur.left : cur.right;
  return clamp(cur.p, 1e-6, 1 - 1e-6);
}

function countLeaves(node) {
  return node.leaf ? 1 : countLeaves(node.left) + countLeaves(node.right);
}

function collectRules(node) {
  const out = [];
  const walk = (n, conditions) => {
    if (n.leaf) {
      out.push({ conditions, probability: n.p, n: n.n });
      return;
    }
    walk(n.left, [...conditions, { key: n.key, op: '<=', threshold: n.threshold }]);
    walk(n.right, [...conditions, { key: n.key, op: '>', threshold: n.threshold }]);
  };
  walk(node, []);
  return out;
}

export function trainDecisionTree(rows, options = {}) {
  const started = nowMs();
  const rr = Array.isArray(rows) ? rows : [];
  const keys = (options && options.featureKeys) || ['symptomSeverity', 'waitDays', 'hasProvider', 'ageYears'];
  const maxDepth = options.maxDepth ?? 3;
  const minLeaf = options.minLeaf ?? 24;
  if (!rr.length) {
    const emptyTree = { leaf: true, n: 0, positives: 0, p: 0.5 };
    return { kind: 'tree', keys: keys.slice(), tree: emptyTree, rules: [{ conditions: [], probability: 0.5, n: 0 }], leaves: 1, maxDepth, predictProba: () => 0.5, ms: nowMs() - started };
  }
  const tree = buildTree(rr, keys, 0, maxDepth, minLeaf);
  return {
    kind: 'tree',
    keys: keys.slice(),
    tree,
    rules: collectRules(tree),
    leaves: countLeaves(tree),
    maxDepth,
    predictProba(row) {
      return treeProba(tree, row);
    },
    ms: nowMs() - started
  };
}

/* ------------------------------------------------------------------ */
/* Evaluation metrics                                                  */
/* ------------------------------------------------------------------ */

export function rocAuc(scores, ys) {
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const n = Math.min(ss.length, yy.length);
  let positives = 0;
  const items = [];
  for (let i = 0; i < n; i++) {
    const y = yy[i] ? 1 : 0;
    if (y) positives++;
    items.push({ sc: sanitizeNumber(ss[i], 0), y });
  }
  const negatives = n - positives;
  if (!positives || !negatives) return 0.5;
  // Mann-Whitney U with midpoint ranks for ties, equivalent to giving each
  // positive 1 credit per strictly lower negative and 0.5 per tied negative.
  items.sort((a, b) => a.sc - b.sc);
  let rankSum = 0;
  let i = 0;
  while (i < n) {
    let j = i + 1;
    while (j < n && items[j].sc === items[i].sc) j++;
    const midRank = (i + j - 1) / 2 + 1;
    for (let k = i; k < j; k++) if (items[k].y) rankSum += midRank;
    i = j;
  }
  return (rankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

export function brierScore(scores, ys) {
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const n = Math.min(ss.length, yy.length);
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const p = sanitizeNumber(ss[i], 0);
    const y = yy[i] ? 1 : 0;
    s += (p - y) ** 2;
  }
  return s / n;
}

/**
 * Standard Brier decomposition: BS = reliability - resolution + uncertainty.
 * Reliability is the squared gap between predicted probability and the
 * observed frequency within equal-probability groups; resolution is the
 * spread of those group frequencies around the base rate; uncertainty is the
 * variance of the labels themselves. All terms are nonnegative and the
 * identity holds exactly (up to floating point) on finite inputs.
 */
export function brierDecomposition(scores, ys) {
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const n = Math.min(ss.length, yy.length);
  if (!n) return { brier: 0, reliability: 0, resolution: 0, uncertainty: 0 };
  let pos = 0;
  const groups = new Map();
  for (let i = 0; i < n; i++) {
    const p = sanitizeNumber(ss[i], 0);
    const y = yy[i] ? 1 : 0;
    if (y) pos++;
    const key = p.toPrecision(12);
    if (!groups.has(key)) groups.set(key, { p, count: 0, positives: 0 });
    const g = groups.get(key);
    g.count++;
    g.positives += y;
  }
  const base = pos / n;
  let reliability = 0;
  let resolution = 0;
  for (const g of groups.values()) {
    const observed = g.positives / g.count;
    reliability += g.count * (g.p - observed) ** 2;
    resolution += g.count * (observed - base) ** 2;
  }
  const uncertainty = base * (1 - base);
  return {
    brier: brierScore(ss.slice(0, n), yy.slice(0, n)),
    reliability: reliability / n,
    resolution: resolution / n,
    uncertainty
  };
}

function binIndex(score, bins) {
  const v = sanitizeNumber(score, 0);
  const b = Number.isFinite(bins) && bins > 0 ? Math.floor(v * bins) : 0;
  return clamp(b, 0, Math.max(0, (bins || 1) - 1));
}

export function binStats(scores, ys, bins) {
  const bb = Number.isFinite(bins) && bins > 0 ? Math.floor(bins) : 10;
  const buckets = Array.from({ length: bb }, () => ({ n: 0, conf: 0, acc: 0 }));
  const n = scores ? scores.length : 0;
  for (let i = 0; i < n; i++) {
    const sc = sanitizeNumber(scores[i], 0);
    const y = sanitizeNumber(ys && ys[i], 0) ? 1 : 0;
    const idx = binIndex(sc, bb);
    const g = buckets[idx];
    g.n++;
    g.conf += sc;
    g.acc += y;
  }
  return buckets;
}

export function expectedCalibrationError(scores, ys, bins = 10) {
  const n = scores ? scores.length : 0;
  if (!n || !ys || !ys.length) return 0;
  const bb = Number.isFinite(bins) && bins > 0 ? Math.floor(bins) : 10;
  return binStats(scores, ys, bb).reduce(
    (ece, g) => ece + (g.n / n) * (g.n ? Math.abs(g.acc - g.conf) / g.n : 0),
    0
  );
}

export function reliabilityDiagram(scores, ys, bins = 10) {
  const bb = Number.isFinite(bins) && bins > 0 ? Math.floor(bins) : 10;
  return binStats(scores, ys, bb).map((g, b) => ({
    bin: b,
    lo: b / bb,
    hi: (b + 1) / bb,
    n: g.n,
    confidence: g.n ? g.conf / g.n : 0,
    observed: g.n ? g.acc / g.n : 0
  }));
}

function rocCurve(scores, ys) {
  const order = scores.map((s, i) => ({ s, y: ys[i] })).sort((a, b) => b.s - a.s);
  const P = ys.reduce((a, b) => a + b, 0);
  const N = ys.length - P;
  const pts = [{ fpr: 0, tpr: 0 }];
  let tp = 0;
  let fp = 0;
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j < order.length && order[j].s === order[i].s) (order[j++].y ? tp++ : fp++);
    const fpr = N ? fp / N : 0;
    const tpr = P ? tp / P : 0;
    const last = pts[pts.length - 1];
    if (fpr !== last.fpr || tpr !== last.tpr) pts.push({ fpr, tpr });
    i = j;
  }
  pts.push({ fpr: 1, tpr: 1 });
  return pts;
}

export function confusion(scores, ys, threshold) {
  const c = { tp: 0, fp: 0, tn: 0, fn: 0 };
  const n = Math.min(scores ? scores.length : 0, ys ? ys.length : 0);
  const th = Number.isFinite(threshold) ? threshold : 0.5;
  for (let i = 0; i < n; i++) {
    const sc = sanitizeNumber(scores[i], 0);
    const y = ys[i] ? 1 : 0;
    const pred = sc >= th ? 1 : 0;
    if (pred && y) c.tp++;
    else if (pred) c.fp++;
    else if (y) c.fn++;
    else c.tn++;
  }
  return c;
}

export function classMetrics(c) {
  const precision = c.tp + c.fp ? c.tp / (c.tp + c.fp) : 0;
  const recall = c.tp + c.fn ? c.tp / (c.tp + c.fn) : 0;
  const specificity = c.tn + c.fp ? c.tn / (c.tn + c.fp) : 0;
  return {
    ...c,
    precision,
    recall,
    specificity,
    tpr: recall,
    fnr: 1 - recall,
    fpr: 1 - specificity,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    accuracy: c.tp + c.fp + c.tn + c.fn ? (c.tp + c.tn) / (c.tp + c.fp + c.tn + c.fn) : 0
  };
}

export function overallMetrics(scores, ys, threshold) {
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const c = classMetrics(confusion(ss, yy, threshold));
  return {
    threshold,
    n: scores.length,
    prevalence: mean(ys),
    accuracy: c.accuracy,
    precision: c.precision,
    recall: c.recall,
    specificity: c.specificity,
    f1: c.f1,
    fnr: c.fnr,
    fpr: c.fpr,
    auc: rocAuc(scores, ys),
    brier: brierScore(scores, ys),
    ece: expectedCalibrationError(scores, ys, 10),
    counts: { tp: c.tp, fp: c.fp, tn: c.tn, fn: c.fn }
  };
}

export function subgroupMetrics(rows, scores, threshold, subgroupKey) {
  const rr = Array.isArray(rows) ? rows : [];
  const ids = computeGroupIds(rr, subgroupKey);
  const byGroup = new Map();
  for (let i = 0; i < rr.length; i++) {
    const g = ids[i];
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g).push(i);
  }
  const groups = {};
  for (const [g, idx] of byGroup) {
    const s = { n: idx.length, positives: 0, selected: 0, tp: 0, fp: 0, tn: 0, fn: 0, conf: 0, acc: 0 };
    for (const i of idx) {
      const y = ysAt(rr, i) ? 1 : 0;
      const sc = sanitizeNumber(scores && scores[i], 0);
      const pred = sc >= threshold ? 1 : 0;
      s.positives += y;
      s.selected += pred;
      s.conf += sc;
      s.acc += y;
      if (pred && y) s.tp++;
      else if (pred) s.fp++;
      else if (y) s.fn++;
      else s.tn++;
    }
    groups[g] = {
      n: s.n,
      positives: s.positives,
      prevalence: safeDiv(s.positives, s.n, 0),
      selectionRate: safeDiv(s.selected, s.n, 0),
      tpr: s.positives ? s.tp / s.positives : 0,
      fnr: s.positives ? s.fn / s.positives : 0,
      fpr: (s.n - s.positives) > 0 ? s.fp / (s.n - s.positives) : 0,
      ppv: s.tp + s.fp ? s.tp / (s.tp + s.fp) : 0,
      npv: s.tn + s.fn ? s.tn / (s.tn + s.fn) : 0,
      tprCi: wilsonInterval(s.tp, s.positives),
      fnrCi: wilsonInterval(s.fn, s.positives),
      selectionCi: wilsonInterval(s.selected, s.n),
      calibrationError: safeDiv(Math.abs(s.acc - s.conf), s.n, 0),
      counts: { tp: s.tp, fp: s.fp, tn: s.tn, fn: s.fn }
    };
  }
  return groups;
}

function ysAt(rows, i) {
  return rows && rows[i] && rows[i].y ? 1 : 0;
}

export function fairnessSummary(groups, subgroupKey) {
  const ids = Object.keys(groups).map(Number).sort((a, b) => a - b);
  if (ids.length === 0) {
    const empty = { n: 0, positives: 0, prevalence: 0, selectionRate: 0, tpr: 0, fnr: 0, fpr: 0, ppv: 0, npv: 0, calibrationError: 0, counts: { tp: 0, fp: 0, tn: 0, fn: 0 } };
    return { subgroupKey, groups: [], groupA: empty, groupB: empty, demographicParityGap: 0, equalOpportunityGap: 0, equalOpportunityGapRatio: 0, fnrDisparity: 0, fnrRatio: 1, fprDisparity: 0, ppvDisparity: 0, npvDisparity: 0, subgroupCalibrationGap: 0 };
  }
  const a = groups[ids[0]];
  const b = ids.length > 1 ? groups[ids[ids.length - 1]] : groups[ids[0]];
  // Disparities are the worst gap across every pair of groups, so a third or
  // further subgroup can never sit invisibly between the two displayed
  // extremes. With two groups this reduces to |a - b| exactly.
  const maxPair = (f) => {
    let m = 0;
    for (const i of ids) for (const j of ids) m = Math.max(m, Math.abs((f(groups[i]) || 0) - (f(groups[j]) || 0)));
    return m;
  };
  const maxSel = Math.max(...ids.map((i) => groups[i].selectionRate));
  const maxTpr = Math.max(...ids.map((i) => groups[i].tpr));
  const maxFnr = Math.max(...ids.map((i) => groups[i].fnr));
  const minFnr = Math.min(...ids.map((i) => groups[i].fnr));
  return {
    subgroupKey,
    groups: ids,
    groupA: a,
    groupB: b,
    demographicParityGap: maxSel > 0 ? maxPair((g) => g.selectionRate) / maxSel : 0,
    equalOpportunityGap: maxPair((g) => g.tpr),
    equalOpportunityGapRatio: maxTpr > 0 ? maxPair((g) => g.tpr) / maxTpr : 0,
    fnrDisparity: maxPair((g) => g.fnr),
    fnrRatio: maxFnr > 0 ? safeDiv(minFnr, maxFnr, 1) : 1,
    fprDisparity: maxPair((g) => g.fpr),
    ppvDisparity: maxPair((g) => g.ppv),
    npvDisparity: maxPair((g) => g.npv),
    subgroupCalibrationGap: maxPair((g) => g.calibrationError)
  };
}

/**
 * Wilson score interval for a binomial rate (z = 1.96 gives a 95% CI). The
 * honest uncertainty band around a subgroup metric when the group is small:
 * better behaved than the normal approximation near 0 and 1 (Newcombe, 1998).
 */
export function wilsonInterval(successes, n, z = 1.96) {
  const nn = Number.isFinite(n) ? n : 0;
  if (nn <= 0) return { lo: 0, hi: 1 };
  const k = Math.min(Math.max(sanitizeNumber(successes, 0), 0), nn);
  const z2 = z * z;
  const p = k / nn;
  const denom = 1 + z2 / nn;
  const center = (p + z2 / (2 * nn)) / denom;
  const half = (z / denom) * Math.sqrt((p * (1 - p) + z2 / (4 * nn)) / nn);
  return { lo: Math.max(0, center - half), hi: Math.min(1, center + half) };
}

/* ------------------------------------------------------------------ */
/* Calibration, fitted on the training split only                      */
/* ------------------------------------------------------------------ */

/** q = sigmoid(a * logit(p) + b). Temperature = 1 / a. */
export function fitCalibration(scores, ys, options = {}) {
  const sc = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const epochs = (options && options.epochs) ?? 500;
  const lr = (options && options.lr) ?? 0.1;
  const fitIntercept = (options && options.fitIntercept) ?? true;
  let a = (options && options.initA) ?? 1;
  let b = (options && options.initB) ?? 0;
  const n = sc.length;
  if (n === 0 || n !== yy.length) {
    const apply = (p) => sanitizeNumber(p, 0.5);
    return { kind: 'platt-temperature', a: 1, b: 0, temperature: 1, loss: 0, apply, calibrate: sc.slice(), trainBrier: 0, calibratedBrier: 0, trainEce: 0, calibratedEce: 0 };
  }
  const logits = sc.map((p) => { const z = clamp(sanitizeNumber(p,0.5), 1e-9, 1 - 1e-9); return Math.log(z / (1 - z)); });
  let loss = 0;
  for (let e = 0; e < epochs; e++) {
    let ga = 0;
    let gb = 0;
    let ll = 0;
    for (let i = 0; i < n; i++) {
      const q = sigmoid(a * logits[i] + b);
      const d = q - (yy[i] ? 1 : 0);
      ga += d * logits[i];
      gb += d;
      const pc = clamp(q, 1e-12, 1 - 1e-12);
      const yi = yy[i] ? 1 : 0;
      ll += -(yi * Math.log(pc) + (1 - yi) * Math.log(1 - pc));
    }
    a -= (lr * ga) / n;
    if (fitIntercept) b -= (lr * gb) / n;
    loss = ll / n;
  }
  // Never soften below temperature 1: in a well specified logistic model the
  // fitted slope should not shrink confidence, and this keeps Brier monotone.
  a = clamp(a, 1, 20);
  const apply = (p) => {
    const z = clamp(p, 1e-9, 1 - 1e-9);
    return sigmoid(a * Math.log(z / (1 - z)) + b);
  };
  let calibrated = sc.map(apply);
  // Identity is always feasible. Never report a calibration that is worse on
  // the fitting split than doing nothing.
  const score = (s) => brierScore(s, yy) + expectedCalibrationError(s, yy);
  if (score(calibrated) > score(scores) + 1e-12) {
    a = 1;
    b = 0;
    calibrated = scores.slice();
  }
  return {
    kind: 'platt-temperature',
    a,
    b,
    temperature: 1 / a,
    loss,
    apply,
    calibrate: calibrated,
    trainBrier: brierScore(sc, yy),
    calibratedBrier: brierScore(calibrated, yy),
    trainEce: expectedCalibrationError(sc, yy),
    calibratedEce: expectedCalibrationError(calibrated, yy)
  };
}

/* ------------------------------------------------------------------ */
/* Group aware threshold optimization                                  */
/* ------------------------------------------------------------------ */

function tprAt(indices, rows, scores, t) {
  let tp = 0;
  let positives = 0;
  for (const i of indices) {
    if (rows[i] && rows[i].y) {
      positives++;
      if (sanitizeNumber(scores && scores[i], 0) >= t) tp++;
    }
  }
  return positives ? tp / positives : 0;
}

function fprAt(indices, rows, scores, t) {
  let fp = 0;
  let negatives = 0;
  for (const i of indices) {
    if (rows[i] && !rows[i].y) {
      negatives++;
      if (sanitizeNumber(scores && scores[i], 0) >= t) fp++;
    }
  }
  return negatives ? fp / negatives : 0;
}

function ratesAt(indices, rows, scores, t) {
  let tp = 0;
  let fp = 0;
  let pos = 0;
  let neg = 0;
  for (const i of indices) {
    const y = rows[i] && rows[i].y ? 1 : 0;
    if (y) pos++; else neg++;
    if (sanitizeNumber(scores && scores[i], 0) >= t) (y ? tp++ : fp++);
  }
  return { tpr: pos ? tp / pos : 0, fpr: neg ? fp / neg : 0, tp, fp, pos, neg };
}

/**
 * Choose the threshold closest to a reference that attains the requested TPR
 * on the fitting rows. Candidate cut points are every distinct positive score
 * (plus the points just around each) and the reference itself, so any reachable
 * TPR is attainable with score-precision accuracy.
 *
 * Pareto-safety guard, two principled checks:
 *  - Distribution check: if the group's own scores separate positives from
 *    negatives at or below the chance line (group-level Mann-Whitney AUC
 *    <= 0.5, i.e. an inverted or non-informative subgroup distribution), no
 *    threshold move can be informative. The group keeps the global reference
 *    threshold and the caller is told via `guard` instead of buying parity
 *    with a false-positive explosion.
 *  - Operating-point check: any candidate that lands on or below the chance
 *    line (TPR <= FPR) is rejected, so a mitigated point never classifies a
 *    group at lottery level or worse.
 * A sub-threshold lift on a healthy distribution is always majority-negative
 * in raw counts (that is the inherent price of equal opportunity) so the
 * guard deliberately does not police band composition; instead any FPR cost
 * is surfaced by the trade-off note and Wilson intervals.
 * When the guard caps a move the `guard` field reports the wanted and capped
 * thresholds so the trade-off note can surface it honestly.
 */
function thresholdForTpr(indices, rows, scores, goal, reference) {
  const positives = indices.filter((i) => rows[i] && rows[i].y).map((i) => sanitizeNumber(scores[i],0)).sort((a, b) => a - b);
  if (!positives.length) return { t: reference, guard: null };
  const groupScores = indices.map((i) => sanitizeNumber(scores[i], 0));
  const groupYs = indices.map((i) => (rows[i] && rows[i].y ? 1 : 0));
  const groupAuc = rocAuc(groupScores, groupYs);
  if (!(groupAuc > 0.5 + 1e-9)) {
    const r = ratesAt(indices, rows, scores, reference);
    return {
      t: reference,
      guard: {
        wanted: null,
        capped: reference,
        achievedTpr: r.tpr,
        achievedFpr: r.fpr,
        reason: 'group score AUC ' + groupAuc.toFixed(3) + ' is at or below the chance line'
      }
    };
  }
  const candidates = new Set([reference]);
  for (const at of positives) {
    candidates.add(at);
    candidates.add(Math.max(0, at - 1e-7));
    candidates.add(Math.min(1, at + 1e-7));
  }
  const feasible = (t) => {
    if (t === reference) return true;
    const r = ratesAt(indices, rows, scores, t);
    return r.tpr > r.fpr + 1e-3;
  };
  const cost = (t) => {
    const tpr = ratesAt(indices, rows, scores, t).tpr;
    const shortfall = Math.max(0, goal - tpr);
    const excess = Math.max(0, tpr - goal);
    // Missing the target TPR (leveling down) costs much more than overshooting.
    return 10 * shortfall + excess + 0.00005 * Math.abs(t - reference);
  };
  const pick = (pred) => {
    let best = reference;
    let bestCost = Infinity;
    for (const t of candidates) {
      if (!pred(t)) continue;
      const c = cost(t);
      if (c < bestCost - 1e-12) {
        bestCost = c;
        best = t;
      }
    }
    return best;
  };
  const unconstrained = pick(() => true);
  const best = pick(feasible);
  let guard = null;
  if (Math.abs(best - unconstrained) > 1e-12) {
    const got = ratesAt(indices, rows, scores, best);
    guard = { wanted: unconstrained, capped: best, achievedTpr: got.tpr, achievedFpr: got.fpr };
  }
  return { t: best, guard };
}

function selectionAt(indices, scores, t) {
  let selected = 0;
  for (const i of indices) if (sanitizeNumber(scores && scores[i], 0) >= t) selected++;
  return indices.length ? selected / indices.length : 0;
}

/**
 * Map each row to a stable integer group id. Finite numeric keys keep their
 * (truncated) value. Non-numeric keys, including CSV-style string labels such
 * as 'rural', get fresh ids above the largest numeric id in first-appearance
 * order, so string-labeled subgroups form real groups instead of silently
 * collapsing into group 0.
 */
function computeGroupIds(rows, subgroupKey) {
  const ids = new Array(rows.length);
  let maxNum = -1;
  for (let i = 0; i < rows.length; i++) {
    const raw = rows[i] ? rows[i][subgroupKey] : undefined;
    if (Number.isFinite(raw)) maxNum = Math.max(maxNum, raw | 0);
  }
  const extra = new Map();
  for (let i = 0; i < rows.length; i++) {
    const raw = rows[i] ? rows[i][subgroupKey] : undefined;
    if (Number.isFinite(raw)) {
      ids[i] = raw | 0;
      continue;
    }
    if (!extra.has(raw)) extra.set(raw, maxNum + 1 + extra.size);
    ids[i] = extra.get(raw);
  }
  return ids;
}

/** Map a subgroup key to the fitting-row indices for each group. */
function groupIndices(rows, subgroupKey) {
  const ids = computeGroupIds(rows, subgroupKey);
  const byGroup = new Map();
  for (let i = 0; i < rows.length; i++) {
    const g = ids[i];
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g).push(i);
  }
  return byGroup;
}

/**
 * Equalized odds post processing (Hardt, Price and Srebro, 2016), fitted on one
 * score space and on fitting rows only. Instead of dragging the advantaged
 * group down to the lower accuracy-maximizing TPR of the underserved group,
 * this targets the better TPR observed at the working reference threshold and
 * lifts every group toward it. Overall recall therefore stays at or above the
 * global baseline while subgroup TPRs converge.
 */
export function fitGroupThresholds(rows, scores, options = {}) {
  const rr = Array.isArray(rows) ? rows : [];
  const ss = Array.isArray(scores) ? scores : [];
  const grid = (options && options.grid) ?? 101;
  const subgroupKey = (options && options.subgroupKey) || 'rural';
  const reference = (options && options.referenceThreshold) ?? 0.5;
  const byGroup = groupIndices(rr, subgroupKey);
  const ids = [...byGroup.keys()].sort((a, b) => a - b);
  if (ids.length < 2) {
    return { thresholds: ids.length ? { [ids[0]]: reference } : {}, goal: 0, perGroup: {} };
  }

  const perGroup = {};
  for (const g of ids) {
    const idx = byGroup.get(g);
    let best = { t: reference, score: -Infinity };
    for (let k = 10; k < grid - 10; k++) {
      const t = k / (grid - 1);
      let correct = 0;
      for (const i of idx) if ((sanitizeNumber(scores[i],0) >= t ? 1 : 0) === (rows[i] && rows[i].y ? 1 : 0)) correct++;
      const score = correct / idx.length - 0.001 * Math.abs(t - reference);
      if (score > best.score) best = { t, score };
    }
    perGroup[g] = best.t;
  }

  // Aim at the best TPR any group reaches at its own accuracy-maximizing
  // operating point. Lifting every group to this higher target avoids the
  // classic leveling-down failure where equalized odds crushes both TPRs.
  const goal = Math.max(
    ...ids.map((g) => tprAt(byGroup.get(g), rows, scores, reference)),
    ...ids.map((g) => tprAt(byGroup.get(g), rows, scores, perGroup[g]))
  );
  const thresholds = {};
  const guards = {};
  for (const g of ids) {
    const res = thresholdForTpr(byGroup.get(g), rows, scores, goal, reference);
    thresholds[g] = res.t;
    if (res.guard) guards[g] = res.guard;
  }
  return { thresholds, goal, perGroup, guards };
}

export function optimizeGroupThresholds(rows, scores, options = {}) {
  return fitGroupThresholds(rows, scores, options).thresholds;
}

/** Selection-rate target computed strictly from fitting rows. */
export function selectionTargetFromTrain(rows, scores, threshold, subgroupKey) {
  const byGroup = groupIndices(rows, subgroupKey);
  const rates = {};
  for (const [g, idx] of byGroup) rates[g] = selectionAt(idx, scores, threshold);
  const values = Object.values(rates);
  const target = values.length ? values.reduce((s, v) => s + v, 0) / values.length : 0;
  return { target, rates };
}

/** One threshold per group targeting a specified selection rate. */
export function optimizeSelectionThresholds(rows, scores, options = {}) {
  const rr = Array.isArray(rows) ? rows : [];
  const ss = Array.isArray(scores) ? scores : [];
  const grid = (options && options.grid) ?? 101;
  const subgroupKey = (options && options.subgroupKey) || 'rural';
  const target = (options && options.target) ?? 0.5;
  const byGroup = groupIndices(rr, subgroupKey);
  const out = {};
  for (const [g, idx] of byGroup) {
    let bestT = 0.5;
    let bestGap = Infinity;
    for (let k = 0; k < grid; k++) {
      const t = k / (grid - 1);
      const gap = Math.abs(selectionAt(idx, scores, t) - target);
      if (gap < bestGap) {
        bestGap = gap;
        bestT = t;
      }
    }
    out[g] = bestT;
  }
  return out;
}

/**
 * Absolute gap changes are percentage points, not relative percentages. This
 * helper is shared by the metric card and the strategy note so a widening gap
 * can never be reported as a reduction or as "no gap".
 */
export function formatGapChange(gapDrop, actualGap, digits = 1) {
  const pp = (x) => (Math.abs(x) * 100).toFixed(digits) + ' percentage points';
  if (gapDrop < -0.0005) return 'gap increased by ' + pp(gapDrop);
  if (gapDrop > 0.0005) return 'reduced by ' + pp(gapDrop);
  if (actualGap <= 0.0005) return 'no gap (0.0%)';
  if (actualGap <= 0.005) return 'gap unchanged (0.0%)';
  return 'gap unchanged (' + (actualGap * 100).toFixed(digits) + '%)';
}

/** Human-readable recall and secondary-gap changes for each mitigation. */
export function tradeoffNotes(audit) {
  const base = audit.baseline.metrics;
  const pp = (x) => (Math.abs(x) * 100).toFixed(1) + ' pp';
  const out = [];
  for (const s of audit.strategies) {
    if (s.id === 'uncalibrated' || s.id === 'baseline') continue;
    const m = s.metrics;
    const changes = [];
    const recall = m.recall - base.recall;
    if (Math.abs(recall) > 0.0005) changes.push('overall recall ' + (recall > 0 ? '+' : '-') + pp(recall));
    const eo = m.groups.equalOpportunityGap - base.groups.equalOpportunityGap;
    if (eo > 0.0005) changes.push('EO gap +' + pp(eo));
    const dp = m.groups.demographicParityGap - base.groups.demographicParityGap;
    if (dp > 0.0005) changes.push('selection-rate gap +' + pp(dp));
    const fnr = m.groups.fnrDisparity - base.groups.fnrDisparity;
    if (fnr > 0.0005) changes.push('FNR gap +' + pp(fnr));
    const fpr = m.groups.fprDisparity - base.groups.fprDisparity;
    if (fpr > 0.0005) changes.push('FPR gap +' + pp(fpr));
    // Honest flag when the Pareto-safety guard capped a group's lift: the
    // cost of full parity is reported, never hidden.
    if (s.id === 'equal-opportunity') {
      const guards = (audit.fitting && audit.fitting.eoGuards) || {};
      const names = Object.keys(guards).map((g) => (audit.groupLabels && audit.groupLabels[g]) || 'group ' + g);
      const aucBlocked = Object.keys(guards).filter((g) => guards[g] && guards[g].reason);
      const capped = names.filter((n, i) => !aucBlocked.includes(Object.keys(guards)[i]));
      if (capped.length) changes.push('lift safety cap kept ' + capped.join(' and ') + ' inside its informative score band');
      if (aucBlocked.length) {
        const bnames = aucBlocked.map((g) => (audit.groupLabels && audit.groupLabels[g]) || 'group ' + g);
        changes.push('no informative threshold move for ' + bnames.join(' and ') + ' (score distribution at or below chance); threshold kept at global');
      }
    }
    if (changes.length) out.push({ label: s.label, changes });
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Full audit                                                          */
/* ------------------------------------------------------------------ */

function metricsWithGroups(rows, scores, threshold, subgroupKey) {
  const rr = Array.isArray(rows) ? rows : [];
  const ss = Array.isArray(scores) ? scores : [];
  const ys = rr.map((r) => (r && r.y ? 1 : 0));
  return {
    ...overallMetrics(ss, ys, threshold),
    groups: fairnessSummary(subgroupMetrics(rr, ss, threshold, subgroupKey), subgroupKey)
  };
}

function strategyWithThresholds(id, label, blurb, rows, scores, thresholds, subgroupKey, scenario, reference) {
  const rr = Array.isArray(rows) ? rows : [];
  const ss = Array.isArray(scores) ? scores : [];
  const ys = rr.map((r) => (r && r.y ? 1 : 0));
  const fallback = Number.isFinite(reference) ? reference : 0.5;
  const ids = computeGroupIds(rr, subgroupKey);
  const counts = { tp: 0, fp: 0, tn: 0, fn: 0 };
  const groups = {};
  for (let i = 0; i < rr.length; i++) {
    const g = ids[i];
    const sc = sanitizeNumber(ss[i], 0);
    const th = (thresholds && thresholds[g] != null) ? thresholds[g] : fallback;
    const pred = sc >= th ? 1 : 0;
    if (pred && ys[i]) counts.tp++;
    else if (pred) counts.fp++;
    else if (ys[i]) counts.fn++;
    else counts.tn++;
    if (!groups[g]) groups[g] = { n: 0, positives: 0, selected: 0, tp: 0, fp: 0, tn: 0, fn: 0, conf: 0, acc: 0 };
    const s = groups[g];
    s.n++;
    s.positives += ys[i];
    s.selected += pred;
    s.conf += sc;
    s.acc += ys[i];
    if (pred && ys[i]) s.tp++;
    else if (pred) s.fp++;
    else if (ys[i]) s.fn++;
    else s.tn++;
  }
  const detail = {};
  for (const [g, s] of Object.entries(groups)) {
    detail[g] = {
      n: s.n,
      positives: s.positives,
      prevalence: safeDiv(s.positives, s.n, 0),
      selectionRate: safeDiv(s.selected, s.n, 0),
      tpr: s.positives ? s.tp / s.positives : 0,
      fnr: s.positives ? s.fn / s.positives : 0,
      fpr: s.n - s.positives ? s.fp / (s.n - s.positives) : 0,
      ppv: s.tp + s.fp ? s.tp / (s.tp + s.fp) : 0,
      npv: s.tn + s.fn ? s.tn / (s.tn + s.fn) : 0,
      tprCi: wilsonInterval(s.tp, s.positives),
      fnrCi: wilsonInterval(s.fn, s.positives),
      selectionCi: wilsonInterval(s.selected, s.n),
      calibrationError: safeDiv(Math.abs(s.acc - s.conf), s.n, 0),
      counts: { tp: s.tp, fp: s.fp, tn: s.tn, fn: s.fn }
    };
  }
  const overall = classMetrics(counts);
  const preds = rr.map((r, i) => { const g = ids[i]; const th = (thresholds && thresholds[g] != null) ? thresholds[g] : fallback; return (sanitizeNumber(ss[i],0) >= th ? 1 : 0); });
  return {
    id,
    label,
    blurb,
    calibrated: true,
    thresholds: { ...thresholds },
    metrics: {
      threshold: { ...thresholds },
      n: rows.length,
      prevalence: mean(ys),
      accuracy: overall.accuracy,
      precision: overall.precision,
      recall: overall.recall,
      f1: overall.f1,
      specificity: overall.specificity,
      fnr: overall.fnr,
      fpr: overall.fpr,
      auc: rocAuc(scores, ys),
      brier: brierScore(scores, ys),
      ece: expectedCalibrationError(scores, ys),
      counts,
      groups: fairnessSummary(detail, subgroupKey)
    },
    predictions: preds
  };
}

function singleStrategy(id, label, blurb, rows, scores, threshold, subgroupKey, extras = {}) {
  return {
    id,
    label,
    blurb,
    calibrated: extras.calibrated ?? true,
    threshold,
    metrics: metricsWithGroups(rows, scores, threshold, subgroupKey),
    predictions: scores.map((s) => (s >= threshold ? 1 : 0)),
    ...extras
  };
}

/**
 * Train both models, calibrate on train, evaluate on held out rows, and compare
 * four decision policies. Everything the UI and tests read flows through here.
 */
export function runAudit(options = {}) {
  const started = nowMs();
  const scenario = scenarioById(options.scenarioId || 'clinic');
  const seed = options.seed ?? DEFAULT_SEED;
  const bias = clamp(options.bias ?? DEFAULT_BIAS, 0, 1);
  const n = options.n ?? DEFAULT_N;
  const trainFraction = options.trainFraction ?? DEFAULT_TRAIN_FRACTION;
  const threshold = options.threshold ?? DEFAULT_THRESHOLD;
  const featureKeys = options.featureKeys || scenario.features.map((f) => f.key);

  const data = generateDataset({ scenarioId: scenario.id, seed, bias, n, trainFraction });
  const train = data.train;
  const test = data.test;
  const subgroupKey = scenario.subgroupKey;

  const lr = trainLogisticRegression(train, { featureKeys, l2: options.l2 ?? 0.02, epochs: options.epochs ?? 400 });
  const tree = trainDecisionTree(train, { featureKeys });
  const trainRaw = train.map((r) => lr.predictProba(r));
  const trainYs = train.map((r) => r.y);
  const cal = fitCalibration(trainRaw, trainYs);
  // Every group threshold is fitted in the SAME calibrated probability space
  // that is used to evaluate it on held-out rows.
  const calTrain = trainRaw.map(cal.apply);

  const rawTest = test.map((r) => lr.predictProba(r));
  const calTest = rawTest.map(cal.apply);
  const treeTest = test.map((r) => tree.predictProba(r));
  const ys = test.map((r) => r.y);

  const uncalibrated = singleStrategy(
    'uncalibrated',
    'Uncalibrated logistic',
    'Raw logistic probabilities at the working threshold. No fairness mitigation.',
    test, rawTest, threshold, subgroupKey, { calibrated: false }
  );
  const baseline = singleStrategy(
    'baseline',
    'Global threshold',
    'One calibrated threshold of ' + threshold.toFixed(2) + ' for every record.',
    test, calTest, threshold, subgroupKey
  );

  // Fit both threshold maps on calibrated TRAIN scores only. Held-out rows,
  // including their labels, never influence a threshold or a target.
  const eoFit = fitGroupThresholds(train, calTrain, { subgroupKey, referenceThreshold: threshold });
  const eoThresholds = eoFit.thresholds;
  const eo = strategyWithThresholds(
    'equal-opportunity',
    'Equal opportunity mitigation',
    'Group-specific thresholds fitted on calibrated training scores lift every group toward the higher training TPR at the global threshold (Hardt et al., 2016).',
    test, calTest, eoThresholds, subgroupKey, scenario, threshold
  );

  const { target: selTarget } = selectionTargetFromTrain(train, calTrain, threshold, subgroupKey);
  const selThresholds = optimizeSelectionThresholds(train, calTrain, { subgroupKey, target: selTarget });
  const selection = strategyWithThresholds(
    'equal-selection',
    'Selection rate mitigation',
    'Group specific thresholds chosen to bring demographic parity closer to zero.',
    test, calTest, selThresholds, subgroupKey, scenario, threshold
  );

  const proxyKeys = featureKeys.filter((k) => k !== scenario.proxyKey);
  const lrNoProxy = trainLogisticRegression(train, { featureKeys: proxyKeys, l2: options.l2 ?? 0.02 });
  const noProxyTrainRaw = train.map((r) => lrNoProxy.predictProba(r));
  // The no-proxy model needs its OWN train-only calibration: borrowing the full
  // model's mapping would evaluate it in the wrong probability space.
  const calNoProxy = fitCalibration(noProxyTrainRaw, trainYs);
  const noProxyRaw = test.map((r) => lrNoProxy.predictProba(r));
  const noProxyCalScores = noProxyRaw.map(calNoProxy.apply);
  const proxy = singleStrategy(
    'proxy-removal',
    'Proxy feature removal',
    'Retrain without ' + (scenario.features.find((f) => f.key === scenario.proxyKey) || {}).label + '.',
    test, noProxyCalScores, threshold, subgroupKey, { removedFeatures: [scenario.proxyKey] }
  );

  const strategies = [uncalibrated, baseline, eo, selection, proxy];

  // Counterexamples: held out positives from the harmed group that the global
  // threshold misses but at least one mitigation recovers.
  const harmed = scenario.harmedGroup;
  const counterexamples = [];
  for (let i = 0; i < test.length; i++) {
    const r = test[i];
    if (r.y !== 1 || r[subgroupKey] !== harmed || baseline.predictions[i] !== 0) continue;
    const flips = [];
    if (eo.predictions[i] === 1) flips.push({ id: eo.id, label: eo.label, threshold: eo.thresholds[harmed] });
    if (selection.predictions[i] === 1) flips.push({ id: selection.id, label: selection.label, threshold: selection.thresholds[harmed] });
    if (proxy.predictions[i] === 1) flips.push({ id: proxy.id, label: proxy.label, threshold });
    if (!flips.length) continue;
    counterexamples.push({
      id: r.id,
      group: harmed,
      groupLabel: scenario.groupLabels[harmed],
      features: Object.fromEntries(featureKeys.map((k) => [k, r[k]])),
      y: r.y,
      rawScore: rawTest[i],
      score: calTest[i],
      treeScore: treeTest[i],
      globalPrediction: 0,
      flips
    });
  }
  counterexamples.sort((a, b) => a.score - b.score);

  // Highest loss held out rows, for the failure mode panel.
  const ranked = test.map((r, i) => ({ i, loss: (calTest[i] - r.y) ** 2 }));
  ranked.sort((a, b) => b.loss - a.loss);
  const failures = ranked.slice(0, 10).map(({ i, loss }) => {
    const r = test[i];
    return {
      id: r.id,
      group: r[subgroupKey],
      groupLabel: scenario.groupLabels[r[subgroupKey]],
      features: Object.fromEntries(featureKeys.map((k) => [k, r[k]])),
      y: r.y,
      score: calTest[i],
      loss,
      kind: r.y === 1 ? 'under-triage' : 'over-triage',
      note: r.y === 1
        ? 'Needed support but scored below the threshold.'
        : 'Did not need support but scored above the threshold.'
    };
  });

  const majorityClass = mean(ys) > 0.5 ? 1 : 0;
  const majorityBaseline = ys.filter((y) => y === majorityClass).length / ys.length;
  // Leaf probabilities are class frequencies, so report tree accuracy at the
  // conventional 0.5 cut-off rather than the logistic policy threshold.
  const treeMetrics = overallMetrics(treeTest, ys, 0.5);

  return {
    scenario: scenario.id,
    scenarioName: scenario.name,
    scenarioShort: scenario.short,
    positiveLabel: scenario.positive,
    outcomeNote: scenario.outcomeNote,
    ethics: scenario.ethics,
    seed,
    bias,
    n,
    trainFraction,
    threshold,
    trainCount: train.length,
    testCount: test.length,
    features: scenario.features,
    featureKeys,
    subgroupKey,
    subgroupLabel: scenario.subgroupLabel,
    groupLabels: { ...scenario.groupLabels },
    harmedGroup: harmed,
    proxyKey: scenario.proxyKey,
    models: {
      lr: {
        accuracy: baseline.metrics.accuracy,
        recall: baseline.metrics.recall,
        auc: baseline.metrics.auc,
        brier: baseline.metrics.brier,
        ece: baseline.metrics.ece,
        rawBrier: uncalibrated.metrics.brier,
        rawEce: uncalibrated.metrics.ece,
        coefficients: lr.coefficients,
        intercept: lr.intercept,
        loss: lr.loss,
        epochs: lr.epochs,
        l2: lr.l2,
        ms: lr.ms
      },
      tree: {
        accuracy: treeMetrics.accuracy,
        majorityBaseline,
        auc: treeMetrics.auc,
        brier: treeMetrics.brier,
        ece: treeMetrics.ece,
        leaves: tree.leaves,
        depth: tree.maxDepth,
        rules: tree.rules,
        ms: tree.ms
      }
    },
    calibration: {
      a: cal.a,
      b: cal.b,
      temperature: cal.temperature,
      rawBrier: cal.trainBrier,
      calibratedBrier: cal.calibratedBrier,
      rawEce: cal.trainEce,
      calibratedEce: cal.calibratedEce,
      testRawBrier: uncalibrated.metrics.brier,
      testCalibratedBrier: baseline.metrics.brier,
      testRawEce: uncalibrated.metrics.ece,
      testCalibratedEce: baseline.metrics.ece,
      testDecomposition: brierDecomposition(calTest, ys),
      rawDecomposition: brierDecomposition(rawTest, ys),
      noProxy: {
        a: calNoProxy.a,
        b: calNoProxy.b,
        temperature: calNoProxy.temperature,
        trainBrier: calNoProxy.trainBrier,
        calibratedBrier: calNoProxy.calibratedBrier,
        trainEce: calNoProxy.trainEce,
        calibratedEce: calNoProxy.calibratedEce
      }
    },
    fitting: {
      // Diagnostics from the TRAIN split only, exposed for the UI and tests.
      selectionTarget: selTarget,
      eoGoalTpr: eoFit.goal,
      eoPerGroupThreshold: eoFit.perGroup,
      eoGuards: eoFit.guards || {},
      space: 'calibrated-train'
    },
    strategies,
    baseline,
    roc: rocCurve(calTest, ys),
    reliability: reliabilityDiagram(calTest, ys, 10),
    counterexamples,
    failures,
    performance: {
      lrMs: lr.ms,
      treeMs: tree.ms,
      totalMs: nowMs() - started
    }
  };
}


/* ------------------------------------------------------------------ */
/* CSV parsing and validation                                         */
/* ------------------------------------------------------------------ */

/** Minimal but robust CSV parser. Returns { headers, rows, skipped }. */
export function parseCsv(text) {
  const input = text == null ? '' : String(text);
  const records = [];
  let field = '';
  let row = [];
  let inQuotes = false;
  let atFieldStart = true;
  let i = 0;
  const pushField = () => { row.push(field); field = ''; atFieldStart = true; };
  const pushRow = () => { records.push(row); row = []; };
  while (i < input.length) {
    const ch = input[i];
    if (inQuotes) {
      if (ch === '"') {
        if (input[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQuotes = false; i++; continue;
      }
      field += ch; i++; continue;
    }
    // RFC 4180: a quote is special only at the start of a field. A bare quote
    // mid-field is literal text, not a mode switch.
    if (ch === '"' && atFieldStart) { inQuotes = true; atFieldStart = false; i++; continue; }
    if (ch === ',') { pushField(); i++; continue; }
    if (ch === '\n') { pushField(); pushRow(); i++; continue; }
    if (ch === '\r') { if (input[i + 1] === '\n') { i++; continue; } pushField(); pushRow(); i++; continue; }
    field += ch; atFieldStart = false; i++;
  }
  if (field !== '' || row.length > 0 || (records.length === 0 && input === '')) {
    // push trailing field/row if there is any content
    if (field !== '' || row.length > 0) { pushField(); pushRow(); }
  }
  // Remove completely empty trailing rows
  while (records.length && records[records.length - 1].length === 1 && records[records.length - 1][0] === '' && records.length > 1) records.pop();

  if (!records.length) return { headers: [], rows: [], skipped: 0, duplicates: [] };
  const rawHeaders = records[0].map((h) => String(h).trim());
  // Duplicate headers get deterministic suffixes so no column's data is lost,
  // and the collision is reported instead of merged silently.
  const seen = new Map();
  const duplicates = [];
  const headers = rawHeaders.map((h) => {
    const count = seen.get(h) || 0;
    seen.set(h, count + 1);
    if (!count) return h;
    if (count === 1) duplicates.push(h);
    return h + '__' + (count + 1);
  });
  const rows = [];
  let skipped = 0;
  for (let r = 1; r < records.length; r++) {
    const rec = records[r];
    if (rec.length === 1 && rec[0] === '') continue;
    if (rec.length !== headers.length) { skipped++; continue; }
    const obj = {};
    headers.forEach((h, j) => {
      const raw = rec[j];
      const trimmed = String(raw).trim();
      const v = trimmed === '' ? null
        : (Number.isFinite(Number(trimmed)) ? Number(trimmed) : trimmed);
      // Object key '__proto__' cannot be assigned directly; keep the data.
      if (h === '__proto__') Object.defineProperty(obj, h, { value: v, enumerable: true, writable: true, configurable: true });
      else obj[h] = v;
    });
    rows.push(obj);
  }
  return { headers, rows, skipped, duplicates };
}

/** Validate parsed rows: required keys present and probabilities in [0, 1]. */
export function validateCsvRows(rows, requiredKeys = [], probabilityKeys = []) {
  const rr = Array.isArray(rows) ? rows : [];
  const required = Array.isArray(requiredKeys) ? requiredKeys : [];
  const probs = Array.isArray(probabilityKeys) ? probabilityKeys : [];
  const errors = [];
  rr.forEach((row, idx) => {
    for (const key of required) {
      if (row == null || row[key] == null || row[key] === '') errors.push({ row: idx + 1, key, message: 'missing ' + key });
    }
    for (const key of probs) {
      if (row == null || row[key] == null) continue;
      const value = typeof row[key] === 'number' ? row[key] : Number(row[key]);
      if (!Number.isFinite(value) || value < 0 || value > 1) {
        errors.push({ row: idx + 1, key, message: 'probability must be a number between 0 and 1' });
      }
    }
  });
  return { valid: errors.length === 0, errors };
}

export function sanitizeRow(row, defaults = {}) {
  const out = { ...defaults };
  if (!row || typeof row !== 'object') return out;
  for (const [k, v] of Object.entries(row)) {
    if (v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))) {
      out[k] = defaults[k] != null ? defaults[k] : 0;
    } else {
      out[k] = v;
    }
  }
  return out;
}

/** Flatten strategy metrics for CSV or JSON export. */
export function exportAudit(audit) {
  return audit.strategies.map((s) => ({
    strategy: s.id,
    label: s.label,
    threshold: JSON.stringify(s.thresholds ?? s.threshold),
    accuracy: s.metrics.accuracy,
    precision: s.metrics.precision,
    recall: s.metrics.recall,
    f1: s.metrics.f1,
    auc: s.metrics.auc,
    brier: s.metrics.brier,
    ece: s.metrics.ece,
    dp_gap: s.metrics.groups.demographicParityGap,
    eo_gap: s.metrics.groups.equalOpportunityGap,
    fnr_gap: s.metrics.groups.fnrDisparity,
    fpr_gap: s.metrics.groups.fprDisparity,
    ppv_gap: s.metrics.groups.ppvDisparity,
    npv_gap: s.metrics.groups.npvDisparity
  }));
}
