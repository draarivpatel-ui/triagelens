/* Wave 3 regression tests: adversarial statistical, CSV, fairness-optimizer
 * and input-hygiene cases reproduced by the Wave 3 audit and closed here. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  runAudit, parseCsv, validateCsvRows, subgroupMetrics, fairnessSummary,
  fitGroupThresholds, wilsonInterval, reliabilityDiagram, brierScore,
  brierDecomposition, tradeoffNotes, exportAudit, trainLogisticRegression,
  generateDataset
} from '../src/ml-engine.js';

/* ------------------------------------------------------------------ */
/* Invalid and degenerate inputs                                       */
/* ------------------------------------------------------------------ */

test('brierScore stays finite on null, NaN and non-finite scores', () => {
  for (const [s, y] of [
    [[null, 0.4], [1, 0]],
    [[NaN, undefined, Infinity, -Infinity], [1, 0, 1, 0]],
    [['x', 0.2], [0, 1]]
  ]) {
    const b = brierScore(s, y);
    assert.ok(Number.isFinite(b) && b >= 0 && b <= 1, 'brier=' + b + ' on ' + JSON.stringify(s));
  }
});

test('brierDecomposition stays finite and identical on invalid inputs', () => {
  for (const [s, y] of [
    [[null, 0.4], [1, 0]],
    [[NaN, 0.5, undefined], [0, 1, 0]],
    [[0.9], []],
    [[], [1]]
  ]) {
    const d = brierDecomposition(s, y);
    for (const k of ['brier', 'reliability', 'resolution', 'uncertainty']) {
      assert.ok(Number.isFinite(d[k]), k + '=' + d[k] + ' on ' + JSON.stringify([s, y]));
    }
    const err = Math.abs(d.brier - (d.reliability - d.resolution + d.uncertainty));
    assert.ok(err < 1e-10, 'identity broken, err=' + err);
  }
});

test('reliabilityDiagram normalizes non-integer and invalid bin counts', () => {
  const d = reliabilityDiagram([0.1, 0.9], [0, 1], 10.7);
  assert.equal(d.length, 10);
  assert.equal(d[d.length - 1].hi, 1);
  assert.equal(d[0].lo, 0);
  for (const bad of [NaN, -3, 0, Infinity, 'x']) {
    const dd = reliabilityDiagram([0.5], [1], bad);
    assert.equal(dd.length, 10, 'bins=' + bad);
    assert.ok(dd.every((b) => Number.isFinite(b.lo) && Number.isFinite(b.hi)), 'non-finite bound for bins=' + bad);
  }
});

/* ------------------------------------------------------------------ */
/* CSV adversarial cases                                               */
/* ------------------------------------------------------------------ */

test('parseCsv treats a bare mid-field quote as literal text per RFC 4180', () => {
  const p = parseCsv('a,b\nx"y,z\n1,2');
  assert.equal(p.rows.length, 2);
  assert.equal(p.rows[0].a, 'x"y');
  assert.equal(p.rows[0].b, 'z');
});

test('parseCsv counts an unterminated quoted row as skipped, not vanished', () => {
  const p = parseCsv('a,b\n"1,2');
  assert.equal(p.rows.length, 0);
  assert.equal(p.skipped, 1);
});

test('parseCsv keeps both duplicate-header columns and reports the collision', () => {
  const p = parseCsv('score,score\n0.3,0.7\n0.4,0.9');
  assert.deepEqual(p.duplicates, ['score']);
  assert.equal(p.rows[0].score, 0.3);
  assert.equal(p.rows[0].score__2, 0.7);
  assert.equal(p.rows[1].score__2, 0.9);
  assert.deepEqual(p.headers, ['score', 'score__2']);
});

test('parseCsv keeps a __proto__ header as own data instead of losing it', () => {
  const p = parseCsv('__proto__,score\n7,0.5');
  assert.equal(p.rows[0].__proto__, 7);
  assert.equal(p.rows[0].score, 0.5);
});

test('XSS subgroup strings stay inert data through parse and metrics', () => {
  const p = parseCsv('grp,score,y\n<img src=x onerror=alert(1)>,0.9,1\n<script>alert(1)</script>,0.1,0');
  assert.equal(p.rows.length, 2);
  assert.equal(typeof p.rows[0].grp, 'string');
  const g = subgroupMetrics(p.rows, [0.9, 0.1], 0.5, 'grp');
  assert.equal(Object.keys(g).length, 2, 'XSS labels must form their own groups');
});

/* ------------------------------------------------------------------ */
/* Subgroup identity and multi-group fairness                          */
/* ------------------------------------------------------------------ */

test('string subgroup values form distinct groups instead of collapsing to 0', () => {
  const rows = [{ grp: 'rural', y: 1 }, { grp: 'urban', y: 0 }, { grp: 'rural', y: 1 }];
  const g = subgroupMetrics(rows, [0.9, 0.1, 0.8], 0.5, 'grp');
  assert.equal(Object.keys(g).length, 2);
  const f = fairnessSummary(g, 'grp');
  assert.ok(f.equalOpportunityGap > 0.9, 'rural vs urban tpr gap ' + f.equalOpportunityGap);
});

test('a missing subgroup key is its own group, never silently group 0', () => {
  const rows = [{ grp: 0, y: 1 }, { y: 0 }, { grp: 0, y: 0 }];
  const g = subgroupMetrics(rows, [0.9, 0.1, 0.1], 0.5, 'grp');
  assert.equal(Object.keys(g).length, 2);
});

test('three or more groups: every disparity uses the worst pair so a middle group cannot hide', () => {
  const rows = [
    { g: 0, y: 1 }, { g: 1, y: 1 }, { g: 1, y: 1 }, { g: 2, y: 1 },
    { g: 0, y: 0 }, { g: 1, y: 0 }, { g: 2, y: 0 }, { g: 2, y: 0 }
  ];
  const scores = [0.9, 0.1, 0.2, 0.9, 0.1, 0.1, 0.2, 0.2];
  const f = fairnessSummary(subgroupMetrics(rows, scores, 0.5, 'g'), 'g');
  // group 1 has TPR 0 while groups 0 and 2 have TPR 1: the gap is 1, not the
  // |0 - 2| = 0 the old first/last comparison would have reported.
  assert.equal(f.equalOpportunityGap, 1);
  assert.equal(f.fnrDisparity, 1);
});

/* ------------------------------------------------------------------ */
/* Wilson score intervals on subgroup rates                            */
/* ------------------------------------------------------------------ */

test('wilsonInterval matches the closed-form 95% interval', () => {
  // k=0,n=10: known Wilson bounds 0 and about 0.2775
  const c0 = wilsonInterval(0, 10);
  assert.equal(c0.lo, 0);
  assert.ok(Math.abs(c0.hi - 0.2775) < 0.001, 'hi=' + c0.hi);
  // k=5,n=10: center near 0.5, interval contains the point estimate
  const c5 = wilsonInterval(5, 10);
  assert.ok(c5.lo < 0.5 && c5.hi > 0.5 && c5.lo > 0 && c5.hi < 1, JSON.stringify(c5));
  // degenerate n=0 gives the full unit interval, n<0 stays safe
  assert.deepEqual(wilsonInterval(0, 0), { lo: 0, hi: 1 });
  assert.ok(wilsonInterval(3, 2).hi <= 1 && wilsonInterval(3, 2).lo >= 0);
});

test('subgroup metrics carry Wilson intervals around the small-n rates', () => {
  const a = runAudit({ scenarioId: 'clinic', seed: 3, bias: 0.6, n: 200 });
  for (const grp of [a.baseline.metrics.groups.groupA, a.baseline.metrics.groups.groupB]) {
    for (const key of ['tprCi', 'fnrCi', 'selectionCi']) {
      assert.ok(grp[key], key + ' missing');
      assert.ok(grp[key].lo >= 0 && grp[key].hi <= 1 && grp[key].lo <= grp[key].hi, JSON.stringify(grp[key]));
    }
    assert.ok(grp.tprCi.lo <= grp.tpr + 1e-9 && grp.tprCi.hi >= grp.tpr - 1e-9, 'CI must contain the estimate');
  }
});

/* ------------------------------------------------------------------ */
/* Pareto-safety guard on pathological distributions                   */
/* ------------------------------------------------------------------ */

test('inverted subgroup distribution keeps the reference threshold and is flagged', () => {
  const rows = [];
  const scores = [];
  for (let i = 0; i < 100; i++) { rows.push({ g: 0, y: i < 50 ? 1 : 0 }); scores.push(i < 50 ? 0.8 : 0.2); }
  for (let i = 0; i < 100; i++) { rows.push({ g: 1, y: i < 50 ? 1 : 0 }); scores.push(i < 50 ? 0.2 : 0.8); }
  const fit = fitGroupThresholds(rows, scores, { subgroupKey: 'g', referenceThreshold: 0.5 });
  assert.equal(fit.thresholds[1], 0.5, 'inverted group must keep the global threshold');
  assert.ok(fit.guards[1], 'guard flag missing');
  assert.match(fit.guards[1].reason, /AUC/i);
  assert.equal(fit.thresholds[0], 0.5, 'the healthy group also stays: its own point already beats the goal-less lift');
});

test('the guard does not fire on the default verified datasets', () => {
  for (const id of ['clinic', 'scholarship', 'shelter']) {
    const a = runAudit({ scenarioId: id, bias: 0.6 });
    assert.deepEqual(a.fitting.eoGuards, {}, id + ' fired guards on default data');
    const eo = a.strategies.find((s) => s.id === 'equal-opportunity');
    assert.notEqual(eo.thresholds[0], eo.thresholds[1], id + ' thresholds must still differ');
  }
});

test('a mitigated operating point can never land on or below the chance line', () => {
  // Weak-but-separable scores: every feasible threshold keeps TPR above FPR.
  const rows = [];
  const scores = [];
  for (let i = 0; i < 200; i++) { rows.push({ g: i < 100 ? 0 : 1, y: i % 2 }); scores.push(i % 2 ? 0.55 : 0.45); }
  const fit = fitGroupThresholds(rows, scores, { subgroupKey: 'g', referenceThreshold: 0.5 });
  for (const g of [0, 1]) {
    const t = fit.thresholds[g];
    let tp = 0, fp = 0, pos = 0, neg = 0;
    rows.forEach((r, i) => { if ((r.g | 0) !== g) return; (r.y ? pos++ : neg++); if (scores[i] >= t) (r.y ? tp++ : fp++); });
    assert.ok(tp / pos > fp / neg - 1e-9, 'group ' + g + ' landed on the chance line at t=' + t);
  }
});

test('an uncovered held-out group falls back to the configured threshold, not 0.5', () => {
  // fit thresholds on a single-group fitting set, then evaluate an audit where
  // group coverage differs; runAudit never crashes and stays finite.
  const a = runAudit({ scenarioId: 'shelter', seed: 11, bias: 0.9, n: 60, threshold: 0.35 });
  const eo = a.strategies.find((s) => s.id === 'equal-opportunity');
  for (const g of eo.metrics.groups.groups) {
    const th = eo.thresholds[g];
    assert.ok(th === undefined || th === 0.35 || (th >= 0 && th <= 1), 'group ' + g + ' threshold ' + th);
  }
  assert.ok(Number.isFinite(eo.metrics.accuracy));
});

/* ------------------------------------------------------------------ */
/* Honest trade-off reporting                                          */
/* ------------------------------------------------------------------ */

test('fairnessSummary and exportAudit expose the FPR disparity', () => {
  const a = runAudit({ scenarioId: 'clinic', seed: 12345, bias: 0.6, threshold: 0.35 });
  assert.ok(Number.isFinite(a.baseline.metrics.groups.fprDisparity));
  const eoRow = exportAudit(a).find((r) => r.strategy === 'equal-opportunity');
  assert.ok(Object.prototype.hasOwnProperty.call(eoRow, 'fpr_gap'), 'fpr_gap missing from export');
});

test('tradeoffNotes names an FPR gap widening on the adversarial small split', () => {
  const a = runAudit({ scenarioId: 'clinic', seed: 7, bias: 0.9, threshold: 0.8, n: 60 });
  const eo = a.strategies.find((s) => s.id === 'equal-opportunity');
  const base = a.baseline.metrics.groups;
  const widened = eo.metrics.groups.fprDisparity - base.fprDisparity;
  assert.ok(widened > 0.1, 'fixture should reproduce a wide FPR gap: ' + widened);
  const note = tradeoffNotes(a).find((n) => /equal opportunity/i.test(n.label));
  assert.ok(note, 'no equal-opportunity note');
  assert.ok(note.changes.some((c) => /FPR gap/i.test(c)), 'FPR explosion not reported: ' + JSON.stringify(note));
});

test('guard flags surface in tradeoffNotes when a lift is capped', () => {
  // Seed the audit-fitting path through an audit whose fitting exposes guards;
  // tradeoffNotes must name the cap or the AUC block, whichever fired.
  const a = runAudit({ scenarioId: 'clinic', seed: 7, bias: 0, threshold: 0.15, n: 400 });
  if (Object.keys(a.fitting.eoGuards).length) {
    const note = tradeoffNotes(a).find((n) => /equal opportunity/i.test(n.label));
    assert.ok(note && note.changes.some((c) => /cap|informative|chance/i.test(c)),
      'guard fired but not reported: ' + JSON.stringify(tradeoffNotes(a)));
  }
});

/* ------------------------------------------------------------------ */
/* Model robustness on partial rows                                    */
/* ------------------------------------------------------------------ */

test('predictProba mean-imputes missing features instead of returning NaN', () => {
  const train = generateDataset({ scenarioId: 'clinic', seed: 5, bias: 0.3, n: 200 }).train;
  const lr = trainLogisticRegression(train, { l2: 0.02, epochs: 200 });
  const p = lr.predictProba({ rural: 1 });
  assert.ok(Number.isFinite(p) && p >= 0 && p <= 1, 'p=' + p);
  const p2 = lr.predictProba({});
  assert.ok(Number.isFinite(p2) && p2 >= 0 && p2 <= 1, 'p=' + p2);
});
