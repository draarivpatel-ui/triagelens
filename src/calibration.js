/* TriageLens calibration, uncertainty and threshold workbench.
 * Pure ES module, zero dependencies, fully deterministic.
 *
 * Second layer of the audit on top of the engine metrics: maximum
 * calibration error, Wilson uncertainty on every subgroup rate,
 * intersectional two-way slices with small-cell warnings, per-group
 * reliability data, a held-out clinical-utility frontier, and a
 * transparent comparator of four threshold policies fitted on TRAIN
 * rows only. Held-out rows are used solely to evaluate and display the
 * resulting operating points; nothing about them is fitted.
 *
 * Every record is synthetic. No row is a real patient, student, or
 * shelter guest.
 */

import {
  sanitizeNumber,
  safeDiv,
  wilsonInterval,
  classMetrics,
  brierScore,
  brierDecomposition,
  binStats,
  expectedCalibrationError,
  reliabilityDiagram,
  computeGroupIds
} from './ml-engine.js';

/** Cells smaller than this carry an automatic reliability warning. */
export const SMALL_CELL_N = 30;
/** Recall floor for the sensitivity guardrail policy. */
export const DEFAULT_FLOOR = 0.9;

function meanLocal(xs) {
  if (!xs || !xs.length) return 0;
  let s = 0;
  for (const x of xs) s += sanitizeNumber(x, 0);
  return s / xs.length;
}

/* ------------------------------------------------------------------ */
/* Calibration error: exact ECE and maximum calibration error          */
/* ------------------------------------------------------------------ */

/**
 * Maximum calibration error: the largest absolute gap between mean
 * predicted confidence and observed frequency over all nonempty equal
 * width bins. Always >= ECE, because ECE is the n-weighted average of
 * the same per-bin gaps.
 */
export function maximumCalibrationError(scores, ys, bins = 10) {
  const bb = Number.isFinite(bins) && bins > 0 ? Math.floor(bins) : 10;
  return binStats(scores, ys, bb).reduce(
    (m, g) => Math.max(m, g.n ? Math.abs(g.acc - g.conf) / g.n : 0),
    0
  );
}

/**
 * Reliability bins with a Wilson 95% interval on each bin's observed
 * event rate, so a sparse bin reads honestly wide instead of precise.
 */
export function reliabilityWithUncertainty(scores, ys, bins = 10) {
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const bb = Number.isFinite(bins) && bins > 0 ? Math.floor(bins) : 10;
  const base = reliabilityDiagram(ss, yy, bb);
  const n = Math.min(ss.length, yy.length);
  // Bin edges are equal width on [0,1]; recover each row's positives per bin
  // straight from the raw pairs so the interval counts match the bins.
  const posInBin = new Array(bb).fill(0);
  for (let i = 0; i < n; i++) {
    const sc = sanitizeNumber(ss[i], 0);
    const b = Math.min(bb - 1, Math.max(0, Math.floor(sc * bb)));
    if (yy[i]) posInBin[b]++;
  }
  return base.map((row, i) => ({
    ...row,
    observedCi: wilsonInterval(posInBin[i], row.n)
  }));
}

/** Reliability diagram data computed separately inside each subgroup. */
export function groupReliability(scores, ys, ids, bins = 10) {
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const ii = Array.isArray(ids) ? ids : [];
  const n = Math.min(ss.length, yy.length, ii.length);
  const byGroup = new Map();
  for (let i = 0; i < n; i++) {
    const g = ii[i];
    if (!byGroup.has(g)) byGroup.set(g, { scores: [], ys: [] });
    const e = byGroup.get(g);
    e.scores.push(ss[i]);
    e.ys.push(yy[i]);
  }
  const out = {};
  for (const [g, e] of byGroup) out[g] = reliabilityWithUncertainty(e.scores, e.ys, bins);
  return out;
}

/* ------------------------------------------------------------------ */
/* Intersectional two-way slices                                       */
/* ------------------------------------------------------------------ */

/**
 * A second, honest slice axis per scenario. Each axis reuses a real
 * feature the model already sees, dichotomized at a fixed, stated
 * cut point (never a data-computed median, so train and held-out rows
 * always land in the same cell for the same feature value).
 */
export const SLICE_DEFS = {
  clinic: {
    key: 'hasProvider',
    dimension: 'usual source of care',
    value: (r) => (sanitizeNumber(r && r.hasProvider, 1) >= 0.5 ? 0 : 1),
    labels: { 0: 'has usual care', 1: 'no usual care' }
  },
  scholarship: {
    key: 'gpa',
    dimension: 'grade average band',
    value: (r) => (sanitizeNumber(r && r.gpa, 100) < 70 ? 1 : 0),
    labels: { 0: 'grade average 70 or above', 1: 'grade average below 70' }
  },
  shelter: {
    key: 'transportAccess',
    dimension: 'transport access',
    value: (r) => (sanitizeNumber(r && r.transportAccess, 1) >= 0.5 ? 0 : 1),
    labels: { 0: 'has transport', 1: 'no transport' }
  }
};

export function sliceDefFor(scenarioId) {
  return SLICE_DEFS[scenarioId] || null;
}

/**
 * Intersectional cells: primary subgroup x a second binary axis. Each
 * cell reports the same rate set as the main subgroup audit plus a
 * small-cell flag when n < minN, because a 12-person miss rate is noise,
 * not a finding. Sorting is by group id then slice id, so the same rows
 * always produce the same cell order.
 */
export function sliceMetrics({
  rows, scores, groupIds, sliceIds, threshold,
  groupLabels = {}, sliceLabels = {}, minN = SMALL_CELL_N
} = {}) {
  const rr = Array.isArray(rows) ? rows : [];
  const ss = Array.isArray(scores) ? scores : [];
  const gg = Array.isArray(groupIds) ? groupIds : [];
  const zz = Array.isArray(sliceIds) ? sliceIds : [];
  const n = Math.min(rr.length, ss.length, gg.length, zz.length);
  const th = Number.isFinite(threshold) ? threshold : 0.5;
  const min = Number.isFinite(minN) && minN > 0 ? Math.floor(minN) : SMALL_CELL_N;
  const acc = new Map();
  for (let i = 0; i < n; i++) {
    const g = gg[i];
    const z = zz[i];
    const key = String(g) + '|' + String(z);
    if (!acc.has(key)) {
      acc.set(key, { group: g, slice: z, n: 0, pos: 0, sel: 0, tp: 0, fp: 0, tn: 0, fn: 0, conf: 0 });
    }
    const c = acc.get(key);
    const y = rr[i] && rr[i].y ? 1 : 0;
    const sc = sanitizeNumber(ss[i], 0);
    const pred = sc >= th ? 1 : 0;
    c.n++;
    c.pos += y;
    c.sel += pred;
    c.conf += sc;
    if (pred && y) c.tp++;
    else if (pred) c.fp++;
    else if (y) c.fn++;
    else c.tn++;
  }
  const cells = [...acc.values()].map((c) => {
    const groupLabel = (groupLabels && groupLabels[c.group]) || 'group ' + c.group;
    const sliceLabel = (sliceLabels && sliceLabels[c.slice]) || 'slice ' + c.slice;
    const tpr = c.pos ? c.tp / c.pos : 0;
    const fnr = c.pos ? c.fn / c.pos : 0;
    const fpr = c.n - c.pos ? c.fp / (c.n - c.pos) : 0;
    const ppv = c.tp + c.fp ? c.tp / (c.tp + c.fp) : 0;
    const smallCell = c.n < min;
    return {
      id: c.group + '|' + c.slice,
      group: c.group,
      slice: c.slice,
      groupLabel,
      sliceLabel,
      label: groupLabel + ' + ' + sliceLabel,
      n: c.n,
      positives: c.pos,
      prevalence: safeDiv(c.pos, c.n, 0),
      selectionRate: safeDiv(c.sel, c.n, 0),
      tpr,
      fnr,
      fpr,
      ppv,
      npv: c.tn + c.fn ? c.tn / (c.tn + c.fn) : 0,
      tprCi: wilsonInterval(c.tp, c.pos),
      fnrCi: wilsonInterval(c.fn, c.pos),
      fprCi: wilsonInterval(c.fp, c.n - c.pos),
      ppvCi: wilsonInterval(c.tp, c.tp + c.fp),
      selectionCi: wilsonInterval(c.sel, c.n),
      calibrationError: safeDiv(Math.abs(c.conf - c.pos), c.n, 0),
      smallCell,
      warning: smallCell
        ? 'n=' + c.n + ' is below ' + min + '; treat every rate in this cell as unstable'
        : null,
      counts: { tp: c.tp, fp: c.fp, tn: c.tn, fn: c.fn }
    };
  });
  cells.sort((a, b) => (a.group - b.group) || (a.slice - b.slice));
  return {
    minN: min,
    cells,
    smallCellCount: cells.filter((c) => c.smallCell).length,
    warnings: cells.filter((c) => c.smallCell).map((c) => c.label + ': ' + c.warning)
  };
}

/* ------------------------------------------------------------------ */
/* Threshold policy evaluation                                         */
/* ------------------------------------------------------------------ */

/**
 * Evaluate one policy on a paired score/label set. `threshold` is either
 * a single number or a map from group id to threshold; a missing group
 * falls back to `fallback` (never silently to 0.5).
 */
export function evaluatePolicy(rows, scores, ys, ids, threshold, fallback = 0.5) {
  const rr = Array.isArray(rows) ? rows : [];
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const ii = Array.isArray(ids) ? ids : [];
  const n = Math.min(rr.length, ss.length, yy.length, ii.length);
  const isMap = threshold !== null && typeof threshold === 'object';
  const fb = Number.isFinite(fallback) ? fallback : 0.5;
  const groups = new Map();
  const counts = { tp: 0, fp: 0, tn: 0, fn: 0 };
  let confSum = 0;
  let confSel = 0;
  let sel = 0;
  for (let i = 0; i < n; i++) {
    const g = ii[i];
    const t = isMap
      ? (Number.isFinite(threshold[g]) ? threshold[g] : fb)
      : (Number.isFinite(threshold) ? threshold : fb);
    const sc = sanitizeNumber(ss[i], 0);
    const y = yy[i] ? 1 : 0;
    const pred = sc >= t ? 1 : 0;
    confSum += sc;
    confSel += pred ? sc : 0;
    sel += pred;
    if (pred && y) counts.tp++;
    else if (pred) counts.fp++;
    else if (y) counts.fn++;
    else counts.tn++;
    if (!groups.has(g)) groups.set(g, { n: 0, pos: 0, sel: 0, tp: 0, fp: 0, tn: 0, fn: 0, conf: 0 });
    const c = groups.get(g);
    c.n++;
    c.pos += y;
    c.sel += pred;
    c.conf += sc;
    if (pred && y) c.tp++;
    else if (pred) c.fp++;
    else if (y) c.fn++;
    else c.tn++;
  }
  const detail = {};
  for (const [g, c] of groups) {
    detail[g] = {
      n: c.n,
      positives: c.pos,
      prevalence: safeDiv(c.pos, c.n, 0),
      selectionRate: safeDiv(c.sel, c.n, 0),
      tpr: c.pos ? c.tp / c.pos : 0,
      fnr: c.pos ? c.fn / c.pos : 0,
      fpr: c.n - c.pos ? c.fp / (c.n - c.pos) : 0,
      ppv: c.tp + c.fp ? c.tp / (c.tp + c.fp) : 0,
      npv: c.tn + c.fn ? c.tn / (c.tn + c.fn) : 0,
      tprCi: wilsonInterval(c.tp, c.pos),
      fnrCi: wilsonInterval(c.fn, c.pos),
      fprCi: wilsonInterval(c.fp, c.n - c.pos),
      ppvCi: wilsonInterval(c.tp, c.tp + c.fp),
      selectionCi: wilsonInterval(c.sel, c.n),
      calibrationError: safeDiv(Math.abs(c.conf - c.pos), c.n, 0),
      counts: { tp: c.tp, fp: c.fp, tn: c.tn, fn: c.fn }
    };
  }
  const gids = Object.keys(detail);
  const maxPair = (f) => {
    let m = 0;
    for (const a of gids) for (const b of gids) m = Math.max(m, Math.abs((f(detail[a]) || 0) - (f(detail[b]) || 0)));
    return m;
  };
  const overall = classMetrics(counts);
  const meanConfidence = n ? confSum / n : 0;
  return {
    n,
    positives: counts.tp + counts.fn,
    counts,
    accuracy: overall.accuracy,
    precision: overall.precision,
    recall: overall.recall,
    specificity: overall.specificity,
    f1: overall.f1,
    tpr: overall.tpr,
    fnr: overall.fnr,
    fpr: overall.fpr,
    selectionRate: n ? sel / n : 0,
    meanConfidence,
    // Alert-conditional calibration gap: among the rows this policy
    // actually flags, does the model's stated risk match the observed
    // hit rate (|PPV - mean predicted risk of alerted rows|). Unlike
    // |accuracy - pooled mean score|, this genuinely depends on where the
    // threshold sits, so the comparator column is policy-specific.
    decisionGap: sel ? Math.abs(overall.precision - confSel / sel) : 0,
    groups: detail,
    gaps: {
      equalOpportunity: maxPair((g) => g.tpr),
      fnr: maxPair((g) => g.fnr),
      fpr: maxPair((g) => g.fpr),
      ppv: maxPair((g) => g.ppv),
      npv: maxPair((g) => g.npv),
      demographicParity: maxPair((g) => g.selectionRate)
    }
  };
}

/* ------------------------------------------------------------------ */
/* Threshold candidates and fitting on the train split only            */
/* ------------------------------------------------------------------ */

/** Sorted distinct candidate thresholds: every reachable operating point. */
export function thresholdCandidates(scores) {
  const ss = Array.isArray(scores) ? scores : [];
  const set = new Set();
  for (const s of ss) {
    const v = sanitizeNumber(s, 0);
    if (Number.isFinite(v) && v >= 0 && v <= 1) set.add(v);
  }
  set.add(0);
  set.add(1);
  return [...set].sort((a, b) => a - b);
}

function recallAt(scores, ys, t) {
  let tp = 0;
  let pos = 0;
  const n = Math.min(scores.length, ys.length);
  for (let i = 0; i < n; i++) {
    if (ys[i]) {
      pos++;
      if (sanitizeNumber(scores[i], 0) >= t) tp++;
    }
  }
  return pos ? tp / pos : 0;
}

/**
 * Sensitivity-floor guardrail: the highest threshold whose fitting
 * recall still meets the floor, which is the minimum alert volume that
 * satisfies the clinical safety constraint. `achievable` reports whether
 * any threshold reaches the floor on the fitting rows (always true when
 * positives exist, since t=0 recalls everything).
 */
export function fitSensitivityFloor(scores, ys, floor = DEFAULT_FLOOR) {
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const target = Number.isFinite(floor) ? floor : DEFAULT_FLOOR;
  const candidates = thresholdCandidates(ss);
  let best = null;
  for (const t of candidates) {
    if (recallAt(ss, yy, t) >= target - 1e-12) best = t;
  }
  if (best === null) return { threshold: 0, floor: target, recall: 0, achievable: false };
  return { threshold: best, floor: target, recall: recallAt(ss, yy, best), achievable: true };
}

/**
 * Youden's J optimal: the single threshold maximizing TPR - FPR on the
 * fitting rows. Ties break toward the reference threshold, then toward
 * the lower candidate, so the result is fully deterministic.
 */
export function fitYoudenThreshold(scores, ys, reference = 0.5) {
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const candidates = thresholdCandidates(ss);
  const ref = Number.isFinite(reference) ? reference : 0.5;
  let best = candidates[0];
  let bestJ = -Infinity;
  let bestDist = Infinity;
  const n = Math.min(ss.length, yy.length);
  for (const t of candidates) {
    let tp = 0, fp = 0, pos = 0, neg = 0;
    for (let i = 0; i < n; i++) {
      const y = yy[i] ? 1 : 0;
      if (sanitizeNumber(ss[i], 0) >= t) (y ? tp++ : fp++);
      if (y) pos++; else neg++;
    }
    const tpr = pos ? tp / pos : 0;
    const fpr = neg ? fp / neg : 0;
    const j = tpr - fpr;
    const dist = Math.abs(t - ref);
    if (j > bestJ + 1e-12 || (Math.abs(j - bestJ) <= 1e-12 && dist < bestDist - 1e-12)) {
      bestJ = j;
      bestDist = dist;
      best = t;
    }
  }
  let tp = 0, fp = 0, pos = 0, neg = 0;
  for (let i = 0; i < n; i++) {
    const y = yy[i] ? 1 : 0;
    if (sanitizeNumber(ss[i], 0) >= best) (y ? tp++ : fp++);
    if (y) pos++; else neg++;
  }
  return {
    threshold: best,
    j: bestJ,
    tpr: pos ? tp / pos : 0,
    fpr: neg ? fp / neg : 0,
    candidates: candidates.length
  };
}

/**
 * Subgroup-calibrated (rate-matched) thresholds: for each group, the
 * threshold whose fitting selection rate is closest to that group's own
 * fitting prevalence. Every group alerts in proportion to how often its
 * members actually need support on the fitting rows.
 */
export function fitRateMatchThresholds(rows, scores, subgroupKey, reference = 0.5) {
  const rr = Array.isArray(rows) ? rows : [];
  const ss = Array.isArray(scores) ? scores : [];
  const ids = computeGroupIds(rr, subgroupKey);
  const ref = Number.isFinite(reference) ? reference : 0.5;
  const byGroup = new Map();
  for (let i = 0; i < Math.min(rr.length, ss.length); i++) {
    const g = ids[i];
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g).push(i);
  }
  const thresholds = {};
  const fitDetail = {};
  for (const [g, idx] of byGroup) {
    const gScores = idx.map((i) => ss[i]);
    const gYs = idx.map((i) => (rr[i] && rr[i].y ? 1 : 0));
    const prevalence = meanLocal(gYs);
    const candidates = thresholdCandidates(gScores);
    let best = ref;
    let bestGap = Infinity;
    let bestDist = Infinity;
    for (const t of candidates) {
      let sel = 0;
      for (const s of gScores) if (sanitizeNumber(s, 0) >= t) sel++;
      const rate = idx.length ? sel / idx.length : 0;
      const gap = Math.abs(rate - prevalence);
      const dist = Math.abs(t - ref);
      if (gap < bestGap - 1e-12 || (Math.abs(gap - bestGap) <= 1e-12 && dist < bestDist - 1e-12)) {
        bestGap = gap;
        bestDist = dist;
        best = t;
      }
    }
    let sel = 0;
    for (const s of gScores) if (sanitizeNumber(s, 0) >= best) sel++;
    thresholds[g] = best;
    fitDetail[g] = { prevalence, selectionRate: idx.length ? sel / idx.length : 0, gap: bestGap, n: idx.length };
  }
  return { thresholds, fitDetail };
}

/* ------------------------------------------------------------------ */
/* Held-out clinical-utility frontier                                  */
/* ------------------------------------------------------------------ */

/**
 * The achievable frontier on the evaluation rows: for every distinct
 * candidate threshold, the pooled operating point plus each subgroup's
 * own point. Missed cases (FNR) and alert volume (selection rate) trade
 * monotonically, so every point is on the Pareto front of that
 * two-objective trade-off; the curve IS the frontier. The per-group
 * curves show where equalized odds fails: at one pooled alert volume
 * the groups sit at different missed-case rates.
 */
export function thresholdFrontier(scores, ys, ids) {
  const ss = Array.isArray(scores) ? scores : [];
  const yy = Array.isArray(ys) ? ys : [];
  const ii = Array.isArray(ids) ? ids : [];
  const n = Math.min(ss.length, yy.length, ii.length);
  const candidates = thresholdCandidates(ss);
  const groupSet = new Set();
  for (let i = 0; i < n; i++) groupSet.add(ii[i]);
  const groupIds = [...groupSet].sort((a, b) => a - b);
  const points = [];
  const byGroup = {};
  for (const g of groupIds) byGroup[g] = [];
  for (const t of candidates) {
    const pool = { tp: 0, fp: 0, tn: 0, fn: 0, sel: 0 };
    const per = new Map(groupIds.map((g) => [g, { tp: 0, fp: 0, tn: 0, fn: 0, sel: 0, n: 0, pos: 0, neg: 0 }]));
    for (let i = 0; i < n; i++) {
      const y = yy[i] ? 1 : 0;
      const pred = sanitizeNumber(ss[i], 0) >= t ? 1 : 0;
      pool.sel += pred;
      if (pred && y) pool.tp++;
      else if (pred) pool.fp++;
      else if (y) pool.fn++;
      else pool.tn++;
      const c = per.get(ii[i]);
      c.n++;
      c.pos += y;
      c.neg += 1 - y;
      c.sel += pred;
      if (pred && y) c.tp++;
      else if (pred) c.fp++;
      else if (y) c.fn++;
      else c.tn++;
    }
    const total = n || 1;
    const pos = pool.tp + pool.fn;
    const neg = pool.tn + pool.fp;
    const tprs = [];
    for (const [g, c] of per) {
      const tpr = c.pos ? c.tp / c.pos : 0;
      const fpr = c.neg ? c.fp / c.neg : 0;
      tprs.push(tpr);
      byGroup[g].push({ t, tpr, fpr, fnr: 1 - tpr, sel: c.n ? c.sel / c.n : 0 });
    }
    const eoGap = tprs.length ? Math.max(...tprs) - Math.min(...tprs) : 0;
    points.push({
      t,
      tpr: pos ? pool.tp / pos : 0,
      fnr: pos ? pool.fn / pos : 0,
      fpr: neg ? pool.fp / neg : 0,
      sel: pool.sel / total,
      tp: pool.tp,
      fp: pool.fp,
      tn: pool.tn,
      fn: pool.fn,
      eoGap
    });
  }
  return { points, byGroup, groupIds, candidates };
}

/* ------------------------------------------------------------------ */
/* The four-policy comparator                                          */
/* ------------------------------------------------------------------ */

/**
 * Compare four decision policies on the same held-out rows. Fits read
 * train rows only; the table shows the exact cost of each policy in
 * missed positives and total alert volume so no trade-off is hidden.
 */
export function buildComparator({
  trainRows, trainScores, testRows, testScores, ysTest, subgroupKey,
  reference = 0.5, floor = DEFAULT_FLOOR
} = {}) {
  const trnR = Array.isArray(trainRows) ? trainRows : [];
  const trnS = Array.isArray(trainScores) ? trainScores : [];
  const tstR = Array.isArray(testRows) ? testRows : [];
  const tstS = Array.isArray(testScores) ? testScores : [];
  const yyT = Array.isArray(ysTest) ? ysTest : [];
  const ref = Number.isFinite(reference) ? reference : 0.5;
  const trainYs = trnR.map((r) => (r && r.y ? 1 : 0));
  const testIds = computeGroupIds(tstR, subgroupKey);

  const floorFit = fitSensitivityFloor(trnS, trainYs, floor);
  const youdenFit = fitYoudenThreshold(trnS, trainYs, ref);
  const rateFit = fitRateMatchThresholds(trnR, trnS, subgroupKey, ref);

  const defs = [
    {
      id: 'global',
      label: 'Single global threshold',
      blurb: 'One user-set threshold for every record.',
      threshold: ref,
      fittedNote: 'user set, not fitted'
    },
    {
      id: 'sensitivity-floor',
      label: 'Sensitivity-floor guardrail',
      blurb: 'Lowest-alert train threshold still reaching ' + Math.round((Number.isFinite(floor) ? floor : DEFAULT_FLOOR) * 100) + '% train recall.',
      threshold: floorFit.threshold,
      fittedNote: 'train recall ' + (floorFit.recall * 100).toFixed(1) + '%' + (floorFit.achievable ? '' : ' (floor unreachable: kept t=0)')
    },
    {
      id: 'subgroup-calibrated',
      label: 'Subgroup-calibrated thresholds',
      blurb: 'Each subgroup alerts at its own train prevalence rate.',
      threshold: rateFit.thresholds,
      fittedNote: 'per-group, fitted on train'
    },
    {
      id: 'youden',
      label: 'Youden-J optimal',
      blurb: 'One threshold maximizing train TPR minus FPR.',
      threshold: youdenFit.threshold,
      fittedNote: 'train J ' + youdenFit.j.toFixed(3)
    }
  ];

  const policies = defs.map((d) => ({
    id: d.id,
    label: d.label,
    blurb: d.blurb,
    threshold: d.threshold,
    fittedNote: d.fittedNote,
    eval: evaluatePolicy(tstR, tstS, yyT, testIds, d.threshold, ref)
  }));
  const base = policies[0].eval;
  for (const p of policies) {
    p.deltas = {
      missed: p.eval.counts.fn - base.counts.fn,
      // Total alert volume change (every flagged record), matching the
      // frontier chart's "alerts per 100" axis, not false positives alone.
      alerts: (p.eval.counts.tp + p.eval.counts.fp) - (base.counts.tp + base.counts.fp),
      eoGap: p.eval.gaps.equalOpportunity - base.gaps.equalOpportunity,
      fnrGap: p.eval.gaps.fnr - base.gaps.fnr,
      ppvGap: p.eval.gaps.ppv - base.gaps.ppv,
      decisionGap: p.eval.decisionGap - base.decisionGap
    };
  }
  return {
    floor: Number.isFinite(floor) ? floor : DEFAULT_FLOOR,
    reference: ref,
    policies,
    fits: {
      sensitivityFloor: floorFit,
      youden: youdenFit,
      rateMatch: rateFit.fitDetail
    }
  };
}

/* ------------------------------------------------------------------ */
/* Full workbench assembly                                             */
/* ------------------------------------------------------------------ */

/**
 * Assemble every workbench block for one audit run. All fitting happens
 * on train rows; all displayed curves and cells describe held-out rows.
 * Plain data only: the result serializes cleanly for export and hashing.
 */
export function buildWorkbench({
  scenarioId, trainRows, trainScores, testRows, testScores, ysTest,
  subgroupKey, groupLabels = {}, reference = 0.5, bins = 10, floor = DEFAULT_FLOOR
} = {}) {
  const trnS = Array.isArray(trainScores) ? trainScores : [];
  const tstS = Array.isArray(testScores) ? testScores : [];
  const tstR = Array.isArray(testRows) ? testRows : [];
  const yyT = Array.isArray(ysTest) ? ysTest : [];
  const bb = Number.isFinite(bins) && bins > 0 ? Math.floor(bins) : 10;
  const testIds = computeGroupIds(tstR, subgroupKey);
  const comparator = buildComparator({
    trainRows, trainScores: trnS, testRows: tstR, testScores: tstS,
    ysTest: yyT, subgroupKey, reference, floor
  });

  const sliceDef = sliceDefFor(scenarioId);
  let slices = { minN: SMALL_CELL_N, cells: [], smallCellCount: 0, warnings: [], second: null };
  if (sliceDef) {
    const sliceIds = tstR.map((r) => sliceDef.value(r));
    const sm = sliceMetrics({
      rows: tstR, scores: tstS, groupIds: testIds, sliceIds,
      threshold: comparator.policies[0].threshold,
      groupLabels, sliceLabels: sliceDef.labels, minN: SMALL_CELL_N
    });
    slices = {
      minN: sm.minN,
      second: { key: sliceDef.key, dimension: sliceDef.dimension, labels: { ...sliceDef.labels } },
      cells: sm.cells,
      smallCellCount: sm.smallCellCount,
      warnings: sm.warnings
    };
  }

  const frontier = thresholdFrontier(tstS, yyT, testIds);
  const policyPoints = comparator.policies.map((p) => ({
    id: p.id,
    label: p.label,
    sel: p.eval.selectionRate,
    fnr: p.eval.fnr,
    fpr: p.eval.fpr,
    tpr: p.eval.tpr
  }));

  return {
    uncertainty: {
      n: Math.min(tstS.length, yyT.length),
      ece: expectedCalibrationError(tstS, yyT, bb),
      mce: maximumCalibrationError(tstS, yyT, bb),
      brier: brierScore(tstS, yyT),
      decomposition: brierDecomposition(tstS, yyT),
      meanConfidence: meanLocal(tstS),
      prevalence: meanLocal(yyT)
    },
    bins: reliabilityWithUncertainty(tstS, yyT, bb),
    groupOverlay: groupReliability(tstS, yyT, testIds, bb),
    slices,
    comparator,
    frontier: {
      points: frontier.points,
      byGroup: frontier.byGroup,
      groupIds: frontier.groupIds,
      policyPoints
    }
  };
}
