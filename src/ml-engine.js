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
  return x < lo ? lo : x > hi ? hi : x;
}

function mean(xs) {
  let s = 0;
  for (const x of xs) s += x;
  return xs.length ? s / xs.length : 0;
}

function stddev(xs) {
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return Math.sqrt(s / Math.max(1, xs.length));
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
  const stats = keys.map((key) => {
    const xs = rows.map((r) => r[key]);
    return { key, mean: mean(xs), sd: stddev(xs) || 1 };
  });
  const z = rows.map((r) => keys.map((key, j) => (r[key] - stats[j].mean) / stats[j].sd));
  return { stats, z };
}

function standardizeOne(row, stats) {
  return stats.map((s) => (row[s.key] - s.mean) / s.sd);
}

/** Full batch gradient descent on L2 regularized log loss. */
export function trainLogisticRegression(rows, options = {}) {
  const started = nowMs();
  const keys = options.featureKeys || rows.keys || ['symptomSeverity', 'waitDays', 'hasProvider', 'ageYears'];
  const l2 = options.l2 ?? 0.02;
  const epochs = options.epochs ?? 120;
  const lr = options.lr ?? 0.35;
  const { stats, z } = standardize(rows, keys);
  const p = keys.length;
  const w = new Array(p).fill(0);
  let b = 0;
  const loss = [];
  const n = rows.length;
  for (let e = 0; e < epochs; e++) {
    const gw = new Array(p).fill(0);
    let gb = 0;
    let logLoss = 0;
    for (let i = 0; i < n; i++) {
      const xi = z[i];
      const yi = rows[i].y;
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
  const keys = options.featureKeys || ['symptomSeverity', 'waitDays', 'hasProvider', 'ageYears'];
  const maxDepth = options.maxDepth ?? 3;
  const minLeaf = options.minLeaf ?? 24;
  const tree = buildTree(rows, keys, 0, maxDepth, minLeaf);
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

function rocAuc(scores, ys) {
  const pos = [];
  const neg = [];
  for (let i = 0; i < ys.length; i++) (ys[i] ? pos : neg).push(scores[i]);
  if (!pos.length || !neg.length) return 0.5;
  pos.sort((a, b) => a - b);
  neg.sort((a, b) => a - b);
  let wins = 0;
  let j = 0;
  for (const s of pos) {
    while (j < neg.length && neg[j] < s) j++;
    let ties = 0;
    while (j + ties < neg.length && neg[j + ties] === s) ties++;
    wins += j + 0.5 * ties;
  }
  return wins / (pos.length * neg.length);
}

function brierScore(scores, ys) {
  if (!scores.length) return 0;
  return scores.reduce((s, p, i) => s + (p - ys[i]) ** 2, 0) / scores.length;
}

function binIndex(score, bins) {
  return clamp(Math.floor(score * bins), 0, bins - 1);
}

function binStats(scores, ys, bins) {
  const buckets = Array.from({ length: bins }, () => ({ n: 0, conf: 0, acc: 0 }));
  for (let i = 0; i < scores.length; i++) {
    const g = buckets[binIndex(scores[i], bins)];
    g.n++;
    g.conf += scores[i];
    g.acc += ys[i];
  }
  return buckets;
}

function expectedCalibrationError(scores, ys, bins = 10) {
  if (!scores.length) return 0;
  return binStats(scores, ys, bins).reduce(
    (ece, g) => ece + (g.n / scores.length) * (g.n ? Math.abs(g.acc - g.conf) / g.n : 0),
    0
  );
}

function reliabilityDiagram(scores, ys, bins = 10) {
  return binStats(scores, ys, bins).map((g, b) => ({
    bin: b,
    lo: b / bins,
    hi: (b + 1) / bins,
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

function confusion(scores, ys, threshold) {
  const c = { tp: 0, fp: 0, tn: 0, fn: 0 };
  for (let i = 0; i < ys.length; i++) {
    const pred = scores[i] >= threshold ? 1 : 0;
    if (pred && ys[i]) c.tp++;
    else if (pred) c.fp++;
    else if (ys[i]) c.fn++;
    else c.tn++;
  }
  return c;
}

function classMetrics(c) {
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

function overallMetrics(scores, ys, threshold) {
  const c = classMetrics(confusion(scores, ys, threshold));
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

function subgroupMetrics(rows, scores, threshold, subgroupKey) {
  const byGroup = new Map();
  for (let i = 0; i < rows.length; i++) {
    const g = rows[i][subgroupKey] | 0;
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g).push(i);
  }
  const groups = {};
  for (const [g, idx] of byGroup) {
    const s = { n: idx.length, positives: 0, selected: 0, tp: 0, fp: 0, tn: 0, fn: 0, conf: 0, acc: 0 };
    for (const i of idx) {
      const y = ysAt(rows, i);
      const pred = scores[i] >= threshold ? 1 : 0;
      s.positives += y;
      s.selected += pred;
      s.conf += scores[i];
      s.acc += y;
      if (pred && y) s.tp++;
      else if (pred) s.fp++;
      else if (y) s.fn++;
      else s.tn++;
    }
    groups[g] = {
      n: s.n,
      positives: s.positives,
      prevalence: s.positives / s.n,
      selectionRate: s.selected / s.n,
      tpr: s.positives ? s.tp / s.positives : 0,
      fnr: s.positives ? s.fn / s.positives : 0,
      fpr: s.n - s.positives ? s.fp / (s.n - s.positives) : 0,
      calibrationError: Math.abs(s.acc - s.conf) / s.n,
      counts: { tp: s.tp, fp: s.fp, tn: s.tn, fn: s.fn }
    };
  }
  return groups;
}

function ysAt(rows, i) {
  return rows[i].y;
}

function fairnessSummary(groups, subgroupKey) {
  const ids = Object.keys(groups).map(Number).sort((a, b) => a - b);
  const a = groups[ids[0]];
  const b = groups[ids[ids.length - 1]];
  const maxSel = Math.max(a.selectionRate, b.selectionRate);
  const maxTpr = Math.max(a.tpr, b.tpr);
  const maxFnr = Math.max(a.fnr, b.fnr);
  return {
    subgroupKey,
    groups: ids,
    groupA: a,
    groupB: b,
    demographicParityGap: maxSel > 0 ? Math.abs(a.selectionRate - b.selectionRate) / maxSel : 0,
    equalOpportunityGap: Math.abs(a.tpr - b.tpr),
    equalOpportunityGapRatio: maxTpr > 0 ? Math.abs(a.tpr - b.tpr) / maxTpr : 0,
    fnrDisparity: Math.abs(a.fnr - b.fnr),
    fnrRatio: maxFnr > 0 ? Math.min(a.fnr, b.fnr) / maxFnr : 1,
    subgroupCalibrationGap: Math.abs(a.calibrationError - b.calibrationError)
  };
}

/* ------------------------------------------------------------------ */
/* Calibration, fitted on the training split only                      */
/* ------------------------------------------------------------------ */

/** q = sigmoid(a * logit(p) + b). Temperature = 1 / a. */
export function fitCalibration(scores, ys, options = {}) {
  const epochs = options.epochs ?? 500;
  const lr = options.lr ?? 0.1;
  const fitIntercept = options.fitIntercept ?? true;
  let a = options.initA ?? 1;
  let b = options.initB ?? 0;
  const n = scores.length;
  const logits = scores.map((p) => Math.log(clamp(p, 1e-9, 1 - 1e-9) / (1 - clamp(p, 1e-9, 1 - 1e-9))));
  let loss = 0;
  for (let e = 0; e < epochs; e++) {
    let ga = 0;
    let gb = 0;
    let ll = 0;
    for (let i = 0; i < n; i++) {
      const q = sigmoid(a * logits[i] + b);
      const d = q - ys[i];
      ga += d * logits[i];
      gb += d;
      const pc = clamp(q, 1e-12, 1 - 1e-12);
      ll += -(ys[i] * Math.log(pc) + (1 - ys[i]) * Math.log(1 - pc));
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
  let calibrated = scores.map(apply);
  // Identity is always feasible. Never report a calibration that is worse on
  // the fitting split than doing nothing.
  const score = (s) => brierScore(s, ys) + expectedCalibrationError(s, ys);
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
    trainBrier: brierScore(scores, ys),
    calibratedBrier: brierScore(calibrated, ys),
    trainEce: expectedCalibrationError(scores, ys),
    calibratedEce: expectedCalibrationError(calibrated, ys)
  };
}

/* ------------------------------------------------------------------ */
/* Group aware threshold optimization                                  */
/* ------------------------------------------------------------------ */

function tprAt(indices, rows, scores, t) {
  let tp = 0;
  let positives = 0;
  for (const i of indices) {
    if (rows[i].y) {
      positives++;
      if (scores[i] >= t) tp++;
    }
  }
  return positives ? tp / positives : 0;
}

function selectionAt(indices, scores, t) {
  let selected = 0;
  for (const i of indices) if (scores[i] >= t) selected++;
  return indices.length ? selected / indices.length : 0;
}

/** Threshold whose TPR is nearest a goal. Ties prefer a threshold near 0.5. */
function nearestTpr(indices, rows, scores, goal, grid) {
  let bestT = 0.5;
  let bestCost = Infinity;
  for (let k = 0; k < grid; k++) {
    const t = k / (grid - 1);
    const cost = Math.abs(tprAt(indices, rows, scores, t) - goal) + 0.00005 * Math.abs(t - 0.5);
    if (cost < bestCost) {
      bestCost = cost;
      bestT = t;
    }
  }
  return bestT;
}

/**
 * Equalized odds post processing (Hardt, Price and Srebro, 2016).
 *
 * Each group keeps the threshold nearest its own accuracy maximizing operating
 * point on the TRAIN split, then the group with lower TPR gets the lowest
 * threshold whose train TPR reaches the better group's TPR. This guarantees the
 * two observed TPR gaps are small instead of merely relabeling the same threshold.
 */
export function optimizeGroupThresholds(rows, scores, options = {}) {
  const grid = options.grid ?? 101;
  const subgroupKey = options.subgroupKey || 'rural';
  const byGroup = new Map();
  for (let i = 0; i < rows.length; i++) {
    const g = rows[i][subgroupKey] | 0;
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g).push(i);
  }
  const ids = [...byGroup.keys()].sort((a, b) => a - b);
  if (ids.length < 2) return { [ids[0]]: 0.5 };

  // Accuracy maximizing threshold per group, restricted to a sane range.
  // Blend in proximity to the global 0.5 operating point on ties so the
  // optimizer does not drift to extremes when a group has few positives.
  const perGroup = {};
  for (const g of ids) {
    const idx = byGroup.get(g);
    let best = { t: 0.5, score: -Infinity };
    for (let k = 10; k < grid - 10; k++) {
      const t = k / (grid - 1);
      let correct = 0;
      for (const i of idx) if ((scores[i] >= t ? 1 : 0) === rows[i].y) correct++;
      const score = correct / idx.length - 0.001 * Math.abs(t - 0.5);
      if (score > best.score) best = { t, score };
    }
    perGroup[g] = best.t;
  }

  let adv = ids[0];
  let dis = ids[ids.length - 1];
  if (tprAt(byGroup.get(adv), rows, scores, perGroup[adv]) <
      tprAt(byGroup.get(dis), rows, scores, perGroup[dis])) {
    [adv, dis] = [dis, adv];
  }
  const goal = tprAt(byGroup.get(adv), rows, scores, perGroup[adv]);
  const disIdx = byGroup.get(dis);
  let chosen = perGroup[dis];
  if (tprAt(disIdx, rows, scores, chosen) < goal - 1e-9) {
    for (let k = grid - 1; k >= 0; k--) {
      const t = k / (grid - 1);
      if (tprAt(disIdx, rows, scores, t) >= goal - 1e-9) {
        chosen = t;
        break;
      }
    }
  }
  const out = {};
  out[adv] = perGroup[adv];
  out[dis] = chosen;
  return out;
}

/** One threshold per group targeting a specified selection rate. */
export function optimizeSelectionThresholds(rows, scores, options = {}) {
  const grid = options.grid ?? 101;
  const subgroupKey = options.subgroupKey || 'rural';
  const target = options.target ?? 0.5;
  const byGroup = new Map();
  for (let i = 0; i < rows.length; i++) {
    const g = rows[i][subgroupKey] | 0;
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g).push(i);
  }
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

/* ------------------------------------------------------------------ */
/* Full audit                                                          */
/* ------------------------------------------------------------------ */

function metricsWithGroups(rows, scores, threshold, subgroupKey) {
  const ys = rows.map((r) => r.y);
  return {
    ...overallMetrics(scores, ys, threshold),
    groups: fairnessSummary(subgroupMetrics(rows, scores, threshold, subgroupKey), subgroupKey)
  };
}

function strategyWithThresholds(id, label, blurb, rows, scores, thresholds, subgroupKey, scenario) {
  const ys = rows.map((r) => r.y);
  const counts = { tp: 0, fp: 0, tn: 0, fn: 0 };
  const groups = {};
  for (let i = 0; i < rows.length; i++) {
    const g = rows[i][subgroupKey] | 0;
    const pred = scores[i] >= thresholds[g] ? 1 : 0;
    if (pred && ys[i]) counts.tp++;
    else if (pred) counts.fp++;
    else if (ys[i]) counts.fn++;
    else counts.tn++;
    if (!groups[g]) groups[g] = { n: 0, positives: 0, selected: 0, tp: 0, fp: 0, tn: 0, fn: 0, conf: 0, acc: 0 };
    const s = groups[g];
    s.n++;
    s.positives += ys[i];
    s.selected += pred;
    s.conf += scores[i];
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
      prevalence: s.positives / s.n,
      selectionRate: s.selected / s.n,
      tpr: s.positives ? s.tp / s.positives : 0,
      fnr: s.positives ? s.fn / s.positives : 0,
      fpr: s.n - s.positives ? s.fp / (s.n - s.positives) : 0,
      calibrationError: Math.abs(s.acc - s.conf) / s.n,
      counts: { tp: s.tp, fp: s.fp, tn: s.tn, fn: s.fn }
    };
  }
  const overall = classMetrics(counts);
  const preds = rows.map((r, i) => (scores[i] >= thresholds[r[subgroupKey] | 0] ? 1 : 0));
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

  const eoThresholds = optimizeGroupThresholds(train, trainRaw, { subgroupKey });
  const eo = strategyWithThresholds(
    'equal-opportunity',
    'Equal opportunity mitigation',
    'Group specific thresholds set so both groups reach the same true positive rate (Hardt et al., 2016).',
    test, calTest, eoThresholds, subgroupKey, scenario
  );

  const selTarget = (baseline.metrics.groups.groupA.selectionRate + baseline.metrics.groups.groupB.selectionRate) / 2;
  const selThresholds = optimizeSelectionThresholds(train, trainRaw, { subgroupKey, target: selTarget });
  const selection = strategyWithThresholds(
    'equal-selection',
    'Selection rate mitigation',
    'Group specific thresholds chosen to bring demographic parity closer to zero.',
    test, calTest, selThresholds, subgroupKey, scenario
  );

  const proxyKeys = featureKeys.filter((k) => k !== scenario.proxyKey);
  const lrNoProxy = trainLogisticRegression(train, { featureKeys: proxyKeys, l2: options.l2 ?? 0.02 });
  const noProxyRaw = test.map((r) => lrNoProxy.predictProba(r));
  const noProxyCalScores = noProxyRaw.map(cal.apply);
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
      testCalibratedEce: baseline.metrics.ece
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
    fnr_gap: s.metrics.groups.fnrDisparity
  }));
}
