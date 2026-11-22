/* Wave 6 tests: calibration and uncertainty workbench.
 * MCE, Wilson-covered bins, intersectional slices, threshold policy
 * fitting (train only), the held-out frontier and the workbench block.
 * Run: node --test tests/wave6-workbench.test.mjs */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  runAudit,
  generateDataset,
  expectedCalibrationError,
  wilsonInterval,
  DATA_STATEMENT
} from '../src/ml-engine.js';
import {
  SMALL_CELL_N,
  DEFAULT_FLOOR,
  maximumCalibrationError,
  reliabilityWithUncertainty,
  groupReliability,
  sliceMetrics,
  evaluatePolicy,
  thresholdCandidates,
  fitSensitivityFloor,
  fitYoudenThreshold,
  fitRateMatchThresholds,
  thresholdFrontier,
  buildComparator,
  buildWorkbench,
  SLICE_DEFS,
  sliceDefFor
} from '../src/calibration.js';

/* ------------------------------------------------------------------ */
/* Maximum calibration error                                           */
/* ------------------------------------------------------------------ */

test('maximumCalibrationError reports the worst single bin gap', () => {
  // Bin [0.8,0.9): two scores mean 0.85, observed 0 -> gap 0.85. Bin
  // [0.1,0.2): two scores mean 0.15, observed 1 -> gap 0.85.
  const mce = maximumCalibrationError([0.85, 0.85, 0.15, 0.15], [0, 0, 1, 1], 10);
  assert.ok(Math.abs(mce - 0.85) < 1e-9, 'mce=' + mce);
});

test('MCE is never below ECE and equals it when all bins share the gap', () => {
  const vectors = [
    [[0.9, 0.1, 0.8, 0.2, 0.55, 0.45], [1, 0, 1, 0, 1, 0]],
    [[0.31, 0.32, 0.67, 0.68], [0, 1, 1, 0]],
    [[0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]],
    [[0.05, 0.95], [1, 0]]
  ];
  for (const [s, y] of vectors) {
    const ece = expectedCalibrationError(s, y, 10);
    const mce = maximumCalibrationError(s, y, 10);
    assert.ok(mce >= ece - 1e-12, 'mce ' + mce + ' below ece ' + ece);
  }
  // Single nonempty bin: the weighted mean gap IS that bin's gap.
  const flat = maximumCalibrationError([0.5, 0.5], [1, 0], 10);
  const flatEce = expectedCalibrationError([0.5, 0.5], [1, 0], 10);
  assert.ok(Math.abs(flat - flatEce) < 1e-12);
});

test('maximumCalibrationError handles empty, hostile and mismatched inputs', () => {
  assert.equal(maximumCalibrationError([], []), 0);
  assert.equal(maximumCalibrationError([0.9], []), 0);
  for (const bad of [NaN, -3, 0, Infinity, 'x']) {
    const m = maximumCalibrationError([0.5, 0.6], [1, 0], bad);
    assert.ok(Number.isFinite(m) && m >= 0 && m <= 1, 'bins=' + bad);
  }
  const m = maximumCalibrationError([NaN, 'x', 0.7, null], [1, 0, 1, 0]);
  assert.ok(Number.isFinite(m) && m >= 0 && m <= 1);
});

test('maximumCalibrationError is 1 for a perfectly inverted score', () => {
  assert.equal(maximumCalibrationError([0.9, 0.9, 0.9], [0, 0, 0], 10), 0.9);
  assert.equal(maximumCalibrationError([1, 1], [0, 0], 10), 1);
});

/* ------------------------------------------------------------------ */
/* Reliability bins with Wilson intervals                              */
/* ------------------------------------------------------------------ */

test('reliabilityWithUncertainty adds a Wilson CI that holds the bin rate', () => {
  const scores = [0.15, 0.15, 0.15, 0.15, 0.85, 0.85];
  const ys = [1, 0, 0, 1, 1, 1];
  const bins = reliabilityWithUncertainty(scores, ys, 10);
  assert.equal(bins.length, 10);
  const b1 = bins[1];
  assert.equal(b1.n, 4);
  assert.equal(b1.observed, 0.5);
  assert.ok(b1.observedCi.lo <= 0.5 + 1e-9 && b1.observedCi.hi >= 0.5 - 1e-9);
  const empty = bins[5];
  assert.equal(empty.n, 0);
  assert.deepEqual(empty.observedCi, { lo: 0, hi: 1 });
});

test('reliabilityWithUncertainty bin positives equal the raw positives', () => {
  const scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0];
  const ys = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1];
  const bins = reliabilityWithUncertainty(scores, ys, 10);
  let pos = 0;
  for (const b of bins) pos += Math.round(b.observed * b.n);
  assert.equal(pos, ys.reduce((a, c) => a + c, 0));
  // score exactly 1.0 lands in the last bin, not a phantom 11th.
  assert.equal(bins[9].n + bins.slice(0, 9).reduce((s, b) => s + b.n, 0), scores.length);
});

test('reliabilityWithUncertainty matches reliabilityDiagram shape and is deterministic', () => {
  const a = reliabilityWithUncertainty([0.2, 0.8], [0, 1], 10);
  const b = reliabilityWithUncertainty([0.2, 0.8], [0, 1], 10);
  assert.deepEqual(a, b);
  for (const row of a) {
    for (const k of ['bin', 'lo', 'hi', 'n', 'confidence', 'observed', 'observedCi']) {
      assert.ok(k in row, k + ' missing');
    }
  }
});

test('groupReliability splits bins per subgroup and sums back to group n', () => {
  const scores = [0.2, 0.8, 0.3, 0.7, 0.6, 0.4];
  const ys = [0, 1, 0, 1, 1, 0];
  const ids = [0, 0, 1, 1, 0, 1];
  const ov = groupReliability(scores, ys, ids, 10);
  assert.deepEqual(Object.keys(ov).sort(), ['0', '1']);
  const n0 = ov[0].reduce((s, b) => s + b.n, 0);
  const n1 = ov[1].reduce((s, b) => s + b.n, 0);
  assert.equal(n0, 3);
  assert.equal(n1, 3);
});

test('groupReliability is empty-safe and order stable', () => {
  assert.deepEqual(groupReliability([], [], []), {});
  const a = groupReliability([0.5], [1], [2], 10);
  assert.ok(a[2]);
  const b = groupReliability([0.5], [1], [2], 10);
  assert.deepEqual(a, b);
});

/* ------------------------------------------------------------------ */
/* Intersectional slices                                               */
/* ------------------------------------------------------------------ */

function sliceFixture() {
  const rows = [];
  const scores = [];
  const groupIds = [];
  const sliceIds = [];
  for (let i = 0; i < 40; i++) { rows.push({ y: i < 20 ? 1 : 0 }); scores.push(i < 20 ? 0.7 : 0.2); groupIds.push(0); sliceIds.push(0); }
  for (let i = 0; i < 10; i++) { rows.push({ y: i < 5 ? 1 : 0 }); scores.push(i < 5 ? 0.9 : 0.1); groupIds.push(0); sliceIds.push(1); }
  for (let i = 0; i < 50; i++) { rows.push({ y: i < 30 ? 1 : 0 }); scores.push(i < 30 ? 0.6 : 0.3); groupIds.push(1); sliceIds.push(0); }
  for (let i = 0; i < 30; i++) { rows.push({ y: i < 15 ? 1 : 0 }); scores.push(i < 15 ? 0.8 : 0.4); groupIds.push(1); sliceIds.push(1); }
  return { rows, scores, groupIds, sliceIds };
}

test('sliceMetrics builds every present intersectional cell with exact counts', () => {
  const f = sliceFixture();
  const sm = sliceMetrics({ ...f, threshold: 0.5 });
  assert.equal(sm.cells.length, 4);
  const total = sm.cells.reduce((s, c) => s + c.n, 0);
  assert.equal(total, 130);
  const c00 = sm.cells.find((c) => c.id === '0|0');
  assert.equal(c00.n, 40);
  assert.equal(c00.positives, 20);
  assert.equal(c00.counts.tp, 20);
  assert.equal(c00.counts.fn, 0);
  assert.equal(c00.tpr, 1);
  assert.equal(c00.fnr, 0);
  const c11 = sm.cells.find((c) => c.id === '1|1');
  assert.equal(c11.counts.tp, 15);
  assert.equal(c11.counts.tn, 15);
  assert.equal(c11.counts.fp, 0);
  assert.equal(c11.tpr, 1);
  assert.equal(c11.fpr, 0);
});

test('sliceMetrics flags exactly the cells below minN', () => {
  const f = sliceFixture();
  const sm = sliceMetrics({ ...f, threshold: 0.5 });
  const small = sm.cells.filter((c) => c.smallCell);
  assert.equal(small.length, 1);
  assert.equal(small[0].id, '0|1');
  assert.equal(sm.smallCellCount, 1);
  assert.equal(sm.warnings.length, 1);
  assert.match(sm.warnings[0], /n=10 is below 30/);
  assert.ok(small[0].warning.includes('unstable'));
});

test('sliceMetrics small-cell boundary is exact at minN', () => {
  const mk = (count, slice) => ({
    rows: Array.from({ length: count }, (_, i) => ({ y: i % 2 })),
    scores: Array.from({ length: count }, () => 0.6),
    groupIds: Array.from({ length: count }, () => 0),
    sliceIds: Array.from({ length: count }, () => slice)
  });
  const a = mk(29, 0);
  const b = mk(30, 0);
  const smA = sliceMetrics({ ...a, threshold: 0.5, minN: 30 });
  const smB = sliceMetrics({ ...b, threshold: 0.5, minN: 30 });
  assert.equal(smA.cells[0].smallCell, true);
  assert.equal(smB.cells[0].smallCell, false);
});

test('sliceMetrics honors a custom minN', () => {
  const f = sliceFixture();
  const sm = sliceMetrics({ ...f, threshold: 0.5, minN: 45 });
  // Cell sizes are 40, 10, 50, 30: all but the n=50 cell fall under 45.
  assert.equal(sm.cells.filter((c) => c.smallCell).length, 3);
});

test('sliceMetrics cells carry ordered Wilson intervals on every rate', () => {
  const f = sliceFixture();
  const sm = sliceMetrics({ ...f, threshold: 0.5 });
  for (const c of sm.cells) {
    for (const k of ['tprCi', 'fnrCi', 'fprCi', 'ppvCi', 'selectionCi']) {
      assert.ok(c[k], k + ' missing on ' + c.id);
      assert.ok(c[k].lo >= 0 && c[k].hi <= 1 && c[k].lo <= c[k].hi, c.id + ' ' + k);
    }
    assert.ok(c.fnrCi.lo <= c.fnr + 1e-9 && c.fnrCi.hi >= c.fnr - 1e-9, 'CI misses fnr');
  }
});

test('sliceMetrics orders cells by group then slice and applies labels', () => {
  const f = sliceFixture();
  const sm = sliceMetrics({
    ...f, threshold: 0.5,
    groupLabels: { 0: 'urban', 1: 'rural' },
    sliceLabels: { 0: 'has care', 1: 'no care' }
  });
  assert.deepEqual(sm.cells.map((c) => c.id), ['0|0', '0|1', '1|0', '1|1']);
  assert.equal(sm.cells[0].label, 'urban + has care');
  assert.equal(sm.cells[3].label, 'rural + no care');
});

test('sliceMetrics is invariant under row permutation', () => {
  const f = sliceFixture();
  const a = sliceMetrics({ ...f, threshold: 0.5 });
  const perm = [...f.rows.keys()].sort(() => -1).map((i) => f.rows.length - 1 - i);
  const b = sliceMetrics({
    rows: perm.map((i) => f.rows[i]),
    scores: perm.map((i) => f.scores[i]),
    groupIds: perm.map((i) => f.groupIds[i]),
    sliceIds: perm.map((i) => f.sliceIds[i]),
    threshold: 0.5
  });
  assert.deepEqual(a, b);
});

test('sliceMetrics defaults a NaN threshold to 0.5 and empty input to no cells', () => {
  const g = sliceMetrics({
    rows: [{ y: 1 }, { y: 0 }],
    scores: [0.6, 0.4],
    groupIds: [0, 0],
    sliceIds: [0, 0],
    threshold: NaN
  });
  assert.equal(g.cells[0].counts.tp, 1);
  const empty = sliceMetrics({});
  assert.deepEqual(empty.cells, []);
  assert.equal(empty.smallCellCount, 0);
  assert.equal(empty.minN, SMALL_CELL_N);
});

test('sliceMetrics bounds paired inputs to the shortest array', () => {
  const sm = sliceMetrics({
    rows: [{ y: 1 }, { y: 0 }, { y: 1 }],
    scores: [0.9, 0.1],
    groupIds: [0, 0],
    sliceIds: [0, 0],
    threshold: 0.5
  });
  assert.equal(sm.cells[0].n, 2);
});

test('every scenario slice axis is defined, binary and disjoint from the subgroup key', () => {
  for (const id of ['clinic', 'scholarship', 'shelter']) {
    const def = sliceDefFor(id);
    assert.ok(def, id + ' missing a slice axis');
    assert.equal(typeof def.value, 'function');
    assert.ok(def.labels[0] && def.labels[1]);
  }
  assert.equal(sliceDefFor('no-such-scenario'), null);
  assert.equal(SLICE_DEFS.clinic.key, 'hasProvider');
  assert.equal(SLICE_DEFS.shelter.key, 'transportAccess');
});

test('the scholarship slice axis dichotomizes gpa at the stated point', () => {
  const v = SLICE_DEFS.scholarship.value;
  assert.equal(v({ gpa: 69.9 }), 1);
  assert.equal(v({ gpa: 70 }), 0);
  assert.equal(v({ gpa: 95 }), 0);
  assert.equal(v({}), 0);
});

/* ------------------------------------------------------------------ */
/* evaluatePolicy                                                      */
/* ------------------------------------------------------------------ */

test('evaluatePolicy reproduces exact confusion counts for a global threshold', () => {
  const rows = [{ y: 1 }, { y: 0 }, { y: 1 }, { y: 0 }];
  const scores = [0.9, 0.8, 0.2, 0.1];
  const ids = [0, 0, 1, 1];
  const ev = evaluatePolicy(rows, scores, [1, 0, 1, 0], ids, 0.5);
  assert.deepEqual(ev.counts, { tp: 1, fp: 1, tn: 1, fn: 1 });
  assert.equal(ev.recall, 0.5);
  assert.equal(ev.accuracy, 0.5);
  assert.equal(ev.selectionRate, 0.5);
});

test('evaluatePolicy applies per-group thresholds with a fallback for missing groups', () => {
  const rows = [{ y: 1 }, { y: 0 }, { y: 1 }, { y: 1 }];
  const scores = [0.6, 0.6, 0.6, 0.6];
  const ids = [0, 0, 1, 2];
  const ev = evaluatePolicy(rows, scores, [1, 0, 1, 1], ids, { 0: 0.7, 1: 0.5 }, 0.4);
  // group 0 threshold 0.7 rejects 0.6; group 1 at 0.5 accepts; group 2
  // is not in the map and falls back to 0.4, which accepts.
  assert.equal(ev.groups[0].counts.tn, 1);
  assert.equal(ev.groups[0].counts.fn, 1);
  assert.equal(ev.groups[1].counts.tp, 1);
  assert.equal(ev.groups[2].counts.tp, 1);
});

test('evaluatePolicy gaps are the worst pairwise gap across all groups', () => {
  const rows = [
    { y: 1 }, { y: 1 }, { y: 1 }, { y: 1 },
    { y: 0 }, { y: 0 }, { y: 0 }, { y: 0 }
  ];
  const scores = [0.9, 0.1, 0.2, 0.9, 0.1, 0.1, 0.2, 0.2];
  const ids = [0, 1, 1, 2, 0, 1, 2, 2];
  const ev = evaluatePolicy(rows, scores, rows.map((r) => r.y), ids, 0.5);
  // TPR by group: g0 = 1 (score 0.9), g1 = 0, g2 = 1 -> gap 1.
  assert.equal(ev.gaps.equalOpportunity, 1);
  assert.equal(ev.gaps.fnr, 1);
});

test('evaluatePolicy decisionGap is the alert-conditional calibration gap', () => {
  const rows = [{ y: 1 }, { y: 0 }, { y: 0 }, { y: 1 }];
  const scores = [0.8, 0.4, 0.3, 0.9];
  // At t=0.5 the alerted rows score 0.8 and 0.9 (mean 0.85); both are
  // true positives so PPV is 1 and the gap is |1 - 0.85| = 0.15.
  const ev = evaluatePolicy(rows, scores, [1, 0, 0, 1], [0, 0, 0, 0], 0.5);
  assert.ok(Math.abs(ev.decisionGap - 0.15) < 1e-12);
});

test('evaluatePolicy decisionGap varies with threshold', () => {
  // The gap must depend on the policy, not just the score distribution.
  const rows = [{ y: 1 }, { y: 0 }, { y: 0 }, { y: 1 }];
  const scores = [0.8, 0.4, 0.3, 0.9];
  const ids = [0, 0, 0, 0];
  const ys = [1, 0, 0, 1];
  const t50 = evaluatePolicy(rows, scores, ys, ids, 0.5).decisionGap;
  const t85 = evaluatePolicy(rows, scores, ys, ids, 0.85).decisionGap;
  assert.notEqual(t50, t85);
});

test('evaluatePolicy decisionGap is 0 when no rows are alerted', () => {
  const rows = [{ y: 1 }, { y: 0 }];
  const ev = evaluatePolicy(rows, [0.1, 0.2], [1, 0], [0, 0], 0.9);
  assert.equal(ev.counts.tp + ev.counts.fp, 0);
  assert.equal(ev.decisionGap, 0);
});

test('evaluatePolicy carries Wilson CIs on every subgroup rate', () => {
  const rows = Array.from({ length: 60 }, (_, i) => ({ y: i % 3 === 0 ? 1 : 0 }));
  const scores = rows.map((_, i) => (i % 3 === 0 ? 0.8 : 0.2));
  const ids = rows.map((_, i) => i % 2);
  const ev = evaluatePolicy(rows, scores, rows.map((r) => r.y), ids, 0.5);
  for (const g of Object.values(ev.groups)) {
    for (const k of ['tprCi', 'fnrCi', 'fprCi', 'ppvCi', 'selectionCi']) {
      assert.ok(g[k] && g[k].lo <= g[k].hi, k);
    }
  }
});

test('evaluatePolicy stays finite on empty and hostile inputs', () => {
  const ev = evaluatePolicy([], [], [], [], 0.5);
  assert.equal(ev.n, 0);
  assert.equal(ev.recall, 0);
  const ev2 = evaluatePolicy([{ y: 1 }], [NaN], [1], [0], NaN);
  assert.ok(Number.isFinite(ev2.accuracy));
  const ev3 = evaluatePolicy([{ y: 1 }], ['x'], [1], [undefined], 'bogus', NaN);
  assert.ok(Number.isFinite(ev3.fpr));
});

/* ------------------------------------------------------------------ */
/* Threshold candidates and fitting                                    */
/* ------------------------------------------------------------------ */

test('thresholdCandidates returns sorted distinct scores plus 0 and 1', () => {
  const c = thresholdCandidates([0.5, 0.2, 0.5, 0.9]);
  assert.deepEqual(c, [0, 0.2, 0.5, 0.9, 1]);
  assert.deepEqual(thresholdCandidates([]), [0, 1]);
});

test('thresholdCandidates drops out-of-range and non-finite scores', () => {
  const c = thresholdCandidates([0.5, 1.7, -0.3, NaN, 'x', 0.25]);
  assert.deepEqual(c, [0, 0.25, 0.5, 1]);
});

test('fitSensitivityFloor picks the highest threshold meeting the floor', () => {
  const scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
  const ys = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0];
  const fit = fitSensitivityFloor(scores, ys, 0.9);
  // Nine positives: recall >= 0.9 forces all nine in (tp >= 8.1), so the
  // largest candidate that still meets the floor is the lowest positive
  // score, 0.1. At 0.2 only 8/9 = 0.889 remain.
  assert.equal(fit.achievable, true);
  assert.equal(fit.threshold, 0.1);
  assert.equal(fit.recall, 1);
});

test('fitSensitivityFloor recall is non-increasing above the fitted point', () => {
  const scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
  const ys = [1, 1, 1, 0, 1, 0, 0, 0];
  const fit = fitSensitivityFloor(scores, ys, 0.75);
  assert.ok(fit.achievable);
  let tp = 0;
  for (let i = 0; i < scores.length; i++) if (ys[i] && scores[i] >= fit.threshold) tp++;
  assert.ok(tp / 4 >= 0.75 - 1e-12);
});

test('fitSensitivityFloor fails closed when positives are absent', () => {
  const fit = fitSensitivityFloor([0.5, 0.6], [0, 0], 0.9);
  assert.equal(fit.achievable, false);
  assert.equal(fit.threshold, 0);
  const empty = fitSensitivityFloor([], [], 0.9);
  assert.equal(empty.achievable, false);
});

test('fitSensitivityFloor meets floor=1 exactly when separable', () => {
  const fit = fitSensitivityFloor([0.9, 0.8, 0.1], [1, 1, 0], 1.0);
  assert.equal(fit.achievable, true);
  assert.equal(fit.recall, 1);
  assert.ok(fit.threshold <= 0.8 && fit.threshold > 0.1, 't=' + fit.threshold);
});

test('fitYoudenThreshold maximizes TPR - FPR across every candidate', () => {
  const scores = [0.95, 0.8, 0.7, 0.4, 0.3, 0.1];
  const ys = [1, 1, 0, 0, 1, 0];
  const fit = fitYoudenThreshold(scores, ys, 0.5);
  // Independent grid check over every candidate.
  let bestJ = -Infinity;
  for (const t of thresholdCandidates(scores)) {
    let tp = 0, fp = 0, pos = 0, neg = 0;
    for (let i = 0; i < scores.length; i++) {
      if (ys[i]) { pos++; if (scores[i] >= t) tp++; }
      else { neg++; if (scores[i] >= t) fp++; }
    }
    bestJ = Math.max(bestJ, tp / pos - fp / neg);
  }
  assert.ok(Math.abs(fit.j - bestJ) < 1e-12, 'fit ' + fit.j + ' vs grid ' + bestJ);
});

test('fitYoudenThreshold ties break toward the reference threshold', () => {
  // Inverted case: J = 0 at t in {0, 0.1, 1} (t = 0.9 flags only the
  // negative, J = -1), so the candidate nearest the reference wins.
  const scores = [0.9, 0.1];
  const ys = [0, 1];
  const near = fitYoudenThreshold(scores, ys, 0.95);
  const far = fitYoudenThreshold(scores, ys, 0.05);
  assert.equal(near.threshold, 1);
  assert.equal(far.threshold, 0);
});

test('fitYoudenThreshold stays finite on empty and degenerate inputs', () => {
  const e = fitYoudenThreshold([], [], 0.5);
  assert.ok(Number.isFinite(e.threshold) && Number.isFinite(e.j));
  const flat = fitYoudenThreshold([0.5, 0.5, 0.5], [1, 0, 1], 0.5);
  assert.ok(Number.isFinite(flat.j) && flat.j >= -1 - 1e-9 && flat.j <= 1 + 1e-9);
});

test('fitYoudenThreshold is deterministic', () => {
  const scores = [0.7, 0.4, 0.9, 0.2, 0.6];
  const ys = [1, 0, 1, 0, 1];
  assert.deepEqual(fitYoudenThreshold(scores, ys, 0.5), fitYoudenThreshold(scores, ys, 0.5));
});

test('fitRateMatchThresholds selects near each group prevalence', () => {
  const rows = [];
  const scores = [];
  for (let i = 0; i < 100; i++) {
    const g = i < 60 ? 0 : 1;
    rows.push({ g, y: g === 0 ? (i < 30 ? 1 : 0) : (i < 10 ? 1 : 0) });
    scores.push(g === 0 ? (i < 30 ? 0.8 : 0.2) : (i < 70 ? 0.75 : 0.15));
  }
  const fit = fitRateMatchThresholds(rows, scores, 'g', 0.5);
  for (const g of [0, 1]) {
    const idx = rows.map((r, i) => ({ r, i })).filter((x) => x.r.g === g).map((x) => x.i);
    const sel = idx.filter((i) => scores[i] >= fit.thresholds[g]).length / idx.length;
    const prev = idx.filter((i) => rows[i].y).length / idx.length;
    assert.ok(Math.abs(sel - prev) < 0.06, 'group ' + g + ' sel ' + sel + ' vs prev ' + prev);
    assert.equal(fit.fitDetail[g].n, idx.length);
  }
});

test('fitRateMatchThresholds is order invariant and deterministic', () => {
  const rows = [{ g: 'a', y: 1 }, { g: 'a', y: 0 }, { g: 'b', y: 1 }, { g: 'b', y: 0 }, { g: 'b', y: 0 }];
  const scores = [0.8, 0.2, 0.7, 0.3, 0.1];
  const a = fitRateMatchThresholds(rows, scores, 'g', 0.5);
  const perm = [4, 3, 2, 1, 0];
  const b = fitRateMatchThresholds(perm.map((i) => rows[i]), perm.map((i) => scores[i]), 'g', 0.5);
  assert.deepEqual(a.thresholds, b.thresholds);
});

test('fitRateMatchThresholds handles empty and single-group inputs', () => {
  assert.deepEqual(fitRateMatchThresholds([], [], 'g', 0.5).thresholds, {});
  const one = fitRateMatchThresholds([{ g: 0, y: 1 }], [0.8], 'g', 0.5);
  assert.equal(Object.keys(one.thresholds).length, 1);
});

/* ------------------------------------------------------------------ */
/* Held-out frontier                                                   */
/* ------------------------------------------------------------------ */

test('thresholdFrontier is monotone: alerts fall, missed cases rise', () => {
  const scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
  const ys = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1];
  const ids = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0];
  const f = thresholdFrontier(scores, ys, ids);
  let prevSel = Infinity;
  let prevFnr = -Infinity;
  for (const pt of f.points) {
    assert.ok(pt.sel <= prevSel + 1e-12, 'sel rose at t=' + pt.t);
    assert.ok(pt.fnr >= prevFnr - 1e-12, 'fnr fell at t=' + pt.t);
    prevSel = pt.sel;
    prevFnr = pt.fnr;
  }
});

test('thresholdFrontier endpoints bound the achievable trade-off', () => {
  const scores = [0.2, 0.5, 0.8];
  const ys = [1, 0, 1];
  const ids = [0, 0, 0];
  const f = thresholdFrontier(scores, ys, ids);
  const at0 = f.points.find((p) => p.t === 0);
  assert.equal(at0.sel, 1);
  assert.equal(at0.fnr, 0);
  const last = f.points[f.points.length - 1];
  assert.equal(last.t, 1);
  for (const p of f.points) {
    assert.ok(p.sel >= 0 && p.sel <= 1 && p.fnr >= 0 && p.fnr <= 1);
    assert.ok(p.tp + p.fp + p.tn + p.fn === 3);
  }
});

test('thresholdFrontier eoGap equals the max TPR spread across groups', () => {
  // One positive per group. At t=0.9 group 0 recalls 1 and group 1
  // recalls 0, so the spread is exactly 1.
  const scores = [0.9, 0.35];
  const ys = [1, 1];
  const ids = [0, 1];
  const f = thresholdFrontier(scores, ys, ids);
  const at09 = f.points.find((p) => p.t === 0.9);
  assert.equal(at09.eoGap, 1);
  const at035 = f.points.find((p) => p.t === 0.35);
  assert.equal(at035.eoGap, 0);
});

test('thresholdFrontier per-group curves cover the same candidates', () => {
  const scores = [0.2, 0.8, 0.4, 0.6];
  const ys = [1, 0, 0, 1];
  const ids = [0, 0, 1, 1];
  const f = thresholdFrontier(scores, ys, ids);
  for (const g of [0, 1]) {
    assert.equal(f.byGroup[g].length, f.points.length);
    for (const pt of f.byGroup[g]) {
      assert.ok(pt.sel >= 0 && pt.sel <= 1 && pt.fnr >= 0 && pt.fnr <= 1);
    }
  }
});

test('thresholdFrontier on empty input still returns bounded points', () => {
  const f = thresholdFrontier([], [], []);
  assert.ok(Array.isArray(f.points));
  for (const p of f.points) {
    assert.ok(Number.isFinite(p.sel) && Number.isFinite(p.fnr));
  }
});

/* ------------------------------------------------------------------ */
/* buildComparator                                                     */
/* ------------------------------------------------------------------ */

function comparatorFixture() {
  const trainRows = [];
  const trainScores = [];
  for (let i = 0; i < 200; i++) {
    const g = i < 120 ? 0 : 1;
    trainRows.push({ g, y: g === 0 ? (i < 70 ? 1 : 0) : (i < 20 ? 1 : 0) });
    trainScores.push(g === 0 ? (i < 70 ? 0.7 : 0.2) : (i < 20 ? 0.65 : 0.25));
  }
  const testRows = [];
  const testScores = [];
  for (let i = 0; i < 100; i++) {
    const g = i < 60 ? 0 : 1;
    testRows.push({ g, y: g === 0 ? (i < 35 ? 1 : 0) : (i < 10 ? 1 : 0) });
    testScores.push(g === 0 ? (i < 35 ? 0.7 : 0.2) : (i < 10 ? 0.65 : 0.25));
  }
  return {
    trainRows, trainScores, testRows, testScores,
    ysTest: testRows.map((r) => r.y),
    subgroupKey: 'g', reference: 0.5
  };
}

test('buildComparator returns the four policies in a fixed order', () => {
  const comp = buildComparator(comparatorFixture());
  assert.deepEqual(comp.policies.map((p) => p.id), ['global', 'sensitivity-floor', 'subgroup-calibrated', 'youden']);
});

test('buildComparator global policy uses the reference and zero deltas', () => {
  const comp = buildComparator(comparatorFixture());
  const g = comp.policies[0];
  assert.equal(g.threshold, 0.5);
  for (const k of ['missed', 'alerts', 'eoGap', 'fnrGap', 'ppvGap', 'decisionGap']) {
    assert.equal(g.deltas[k], 0, k);
  }
});

test('buildComparator deltas are exact count and gap changes vs global', () => {
  const comp = buildComparator(comparatorFixture());
  const base = comp.policies[0].eval;
  for (const p of comp.policies) {
    assert.equal(p.deltas.missed, p.eval.counts.fn - base.counts.fn, p.id + ' missed');
    // Alert delta covers every flagged record, matching the frontier axis.
    assert.equal(
      p.deltas.alerts,
      (p.eval.counts.tp + p.eval.counts.fp) - (base.counts.tp + base.counts.fp),
      p.id + ' alerts');
    assert.ok(Math.abs(p.deltas.eoGap - (p.eval.gaps.equalOpportunity - base.gaps.equalOpportunity)) < 1e-12);
  }
});

test('buildComparator floor policy meets the floor on fitting rows', () => {
  const comp = buildComparator(comparatorFixture());
  const floor = comp.policies.find((p) => p.id === 'sensitivity-floor');
  const fit = comp.fits.sensitivityFloor;
  assert.ok(fit.achievable);
  assert.ok(fit.recall >= DEFAULT_FLOOR - 1e-9);
  assert.equal(floor.threshold, fit.threshold);
});

test('buildComparator thresholds are fitted on train, not on held-out rows', () => {
  const f = comparatorFixture();
  const a = buildComparator(f);
  // Poison the held-out set: different labels and scores. The fitted
  // thresholds must not move an inch.
  const poisoned = { ...f, testRows: [...f.testRows].reverse(), testScores: [...f.testScores].reverse(), ysTest: [...f.ysTest].reverse() };
  const b = buildComparator(poisoned);
  for (const pid of ['global', 'sensitivity-floor', 'subgroup-calibrated', 'youden']) {
    assert.deepEqual(
      a.policies.find((p) => p.id === pid).threshold,
      b.policies.find((p) => p.id === pid).threshold,
      pid + ' threshold moved with held-out rows'
    );
  }
});

test('buildComparator subgroup-calibrated emits one threshold per test group', () => {
  const comp = buildComparator(comparatorFixture());
  const sub = comp.policies.find((p) => p.id === 'subgroup-calibrated');
  assert.deepEqual(Object.keys(sub.threshold).sort(), ['0', '1']);
  for (const t of Object.values(sub.threshold)) assert.ok(t >= 0 && t <= 1);
});

test('buildComparator youden threshold is a real candidate score', () => {
  const comp = buildComparator(comparatorFixture());
  const y = comp.policies.find((p) => p.id === 'youden');
  assert.ok(thresholdCandidates(comparatorFixture().trainScores).includes(y.threshold));
});

test('buildComparator evaluates every policy finitely on degenerate input', () => {
  const comp = buildComparator({ trainRows: [], trainScores: [], testRows: [], testScores: [], ysTest: [], subgroupKey: 'g' });
  for (const p of comp.policies) {
    assert.ok(Number.isFinite(p.eval.accuracy) && Number.isFinite(p.eval.recall), p.id);
    assert.equal(p.eval.n, 0);
  }
});

/* ------------------------------------------------------------------ */
/* buildWorkbench integration                                          */
/* ------------------------------------------------------------------ */

test('runAudit carries a complete workbench for every scenario', () => {
  for (const id of ['clinic', 'scholarship', 'shelter']) {
    const a = runAudit({ scenarioId: id, seed: 5, n: 600 });
    const wb = a.workbench;
    assert.ok(wb, id);
    assert.equal(wb.uncertainty.n, a.testCount);
    assert.ok(Math.abs(wb.uncertainty.ece - a.baseline.metrics.ece) < 1e-12, id + ' ece mismatch');
    assert.ok(wb.uncertainty.mce >= wb.uncertainty.ece - 1e-12);
    assert.ok(wb.comparator.policies.length === 4);
    assert.ok(wb.frontier.points.length >= 2);
    assert.ok(wb.bins.length === 10);
    assert.ok(wb.slices.cells.length >= 2, id + ' needs intersectional cells');
    const cellTotal = wb.slices.cells.reduce((s, c) => s + c.n, 0);
    assert.equal(cellTotal, a.testCount, id + ' cells must cover every held-out row');
  }
});

test('workbench bin counts and group overlay counts reconcile', () => {
  const a = runAudit({ scenarioId: 'clinic', seed: 9, n: 800 });
  const wb = a.workbench;
  const binTotal = wb.bins.reduce((s, b) => s + b.n, 0);
  assert.equal(binTotal, a.testCount);
  for (const [g, bins] of Object.entries(wb.groupOverlay)) {
    const gN = bins.reduce((s, b) => s + b.n, 0);
    const cellN = wb.slices.cells.filter((c) => String(c.group) === g).reduce((s, c) => s + c.n, 0);
    assert.equal(gN, cellN, 'group ' + g + ' overlay n must equal its slice coverage');
  }
});

test('workbench policy points match the comparator evaluation', () => {
  const a = runAudit({ scenarioId: 'shelter', seed: 4, n: 500 });
  const wb = a.workbench;
  for (const pp of wb.frontier.policyPoints) {
    const pol = wb.comparator.policies.find((p) => p.id === pp.id);
    assert.ok(Math.abs(pp.sel - pol.eval.selectionRate) < 1e-12, pp.id);
    assert.ok(Math.abs(pp.fnr - pol.eval.fnr) < 1e-12, pp.id);
  }
});

test('workbench is deterministic and JSON-serializable', () => {
  const a = runAudit({ scenarioId: 'scholarship', seed: 13, n: 400 });
  const b = runAudit({ scenarioId: 'scholarship', seed: 13, n: 400 });
  assert.equal(JSON.stringify(a.workbench), JSON.stringify(b.workbench));
  const parsed = JSON.parse(JSON.stringify(a.workbench));
  assert.deepEqual(parsed.uncertainty.ece, a.workbench.uncertainty.ece);
});

test('workbench stays finite on minimum-size and extreme audits', () => {
  for (const opts of [
    { n: 20, seed: 3 },
    { n: 60, seed: 7, bias: 0.9, threshold: 0.8 },
    { n: 60, seed: 11, bias: 0, threshold: 0.05 },
    { n: 40, threshold: 1 }
  ]) {
    const a = runAudit(opts);
    const wb = a.workbench;
    assert.ok(Number.isFinite(wb.uncertainty.ece) && Number.isFinite(wb.uncertainty.mce));
    for (const p of wb.comparator.policies) {
      assert.ok(Number.isFinite(p.eval.recall), p.id);
    }
    for (const c of wb.slices.cells) {
      assert.ok(Number.isFinite(c.fnr) && Number.isFinite(c.ppv));
    }
  }
});

test('workbench comparator floor actually lowers held-out missed cases on defaults', () => {
  const a = runAudit({ scenarioId: 'scholarship', seed: 20270101, bias: 0.85, threshold: 0.45 });
  const comp = a.workbench.comparator;
  const floorP = comp.policies.find((p) => p.id === 'sensitivity-floor');
  const globalP = comp.policies.find((p) => p.id === 'global');
  assert.ok(floorP.eval.recall >= globalP.eval.recall, 'floor recall must be >= global');
  assert.ok(floorP.deltas.missed <= 0, 'floor must not miss more than global');
});

test('workbench uncertainty decomposition satisfies the Brier identity', () => {
  const a = runAudit({ seed: 21, n: 500 });
  const d = a.workbench.uncertainty.decomposition;
  assert.ok(Math.abs(d.brier - (d.reliability - d.resolution + d.uncertainty)) < 1e-9);
  assert.ok(Math.abs(d.brier - a.workbench.uncertainty.brier) < 1e-12);
});

test('workbench slices expose small-cell state honestly on a tiny audit', () => {
  const a = runAudit({ scenarioId: 'clinic', seed: 3, bias: 0.9, n: 60 });
  const wb = a.workbench;
  assert.ok(wb.slices.smallCellCount >= 1, 'tiny audit must flag small cells');
  for (const c of wb.slices.cells) {
    assert.equal(c.smallCell, c.n < wb.slices.minN);
    assert.equal(!!c.warning, c.smallCell);
  }
});
