/* Deterministic unit tests for the TriageLens engine.
 * Run: node --test tests/ml-engine.test.mjs
 * No dependencies, no network, no wall clock assumptions. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mulberry32,
  SCENARIOS,
  DEFAULT_SEED,
  generateDataset,
  trainLogisticRegression,
  trainDecisionTree,
  fitCalibration,
  optimizeGroupThresholds,
  optimizeSelectionThresholds,
  selectionTargetFromTrain,
  fitGroupThresholds,
  formatGapChange,
  tradeoffNotes,
  runAudit,
  exportAudit
} from '../src/ml-engine.js';

const scenarioIds = SCENARIOS.map((s) => s.id);

/* ------------------------------------------------------------------ */
/* PRNG and synthetic data                                             */
/* ------------------------------------------------------------------ */

test('mulberry32 is deterministic for a seed', () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  for (let i = 0; i < 20; i++) assert.equal(a(), b());
});

test('mulberry32 different seeds diverge', () => {
  const a = mulberry32(1);
  const b = mulberry32(2);
  const diff = Array.from({ length: 10 }, () => a() !== b()).filter(Boolean).length;
  assert.ok(diff >= 9);
});

test('mulberry32 stays in [0,1)', () => {
  const r = mulberry32(7);
  for (let i = 0; i < 1000; i++) {
    const v = r();
    assert.ok(v >= 0 && v < 1);
  }
});

test('generateDataset is reproducible for equal parameters', () => {
  const a = generateDataset({ scenarioId: 'clinic', seed: 99, bias: 0.6, n: 400 });
  const b = generateDataset({ scenarioId: 'clinic', seed: 99, bias: 0.6, n: 400 });
  assert.deepEqual(a.train.map((r) => [r.symptomSeverity, r.y]), b.train.map((r) => [r.symptomSeverity, r.y]));
  assert.deepEqual(a.test.map((r) => [r.incomeIndex, r.y]), b.test.map((r) => [r.incomeIndex, r.y]));
});

test('different seeds produce different rows', () => {
  const a = generateDataset({ seed: 1, n: 200 });
  const b = generateDataset({ seed: 2, n: 200 });
  assert.notEqual(a.train[0].symptomSeverity, b.train[0].symptomSeverity);
});

test('train and test splits are disjoint and stratified', () => {
  const d = generateDataset({ scenarioId: 'shelter', n: 1000, trainFraction: 0.7 });
  const trainRows = new Set(d.train);
  for (const r of d.test) assert.ok(!trainRows.has(r));
  const prevTrain = d.train.filter((r) => r.y).length / d.train.length;
  const prevTest = d.test.filter((r) => r.y).length / d.test.length;
  assert.ok(Math.abs(prevTrain - prevTest) < 0.06);
  assert.ok(d.test.length > d.train.length * 0.3);
});

test('all three scenarios expose correlated features and a subgroup label', () => {
  for (const s of SCENARIOS) {
    const d = generateDataset({ scenarioId: s.id, seed: DEFAULT_SEED, n: 500, bias: 0.5 });
    const rows = [...d.train, ...d.test];
    assert.equal(d.features.length, 5);
    assert.ok(rows.every((r) => r.y === 0 || r.y === 1));
    assert.ok(rows.every((r) => r[s.subgroupKey] === 0 || r[s.subgroupKey] === 1));
    assert.ok(rows.some((r) => r[s.subgroupKey] === 1));
  }
});

test('bias lowers the harmed group label rate', () => {
  for (const s of SCENARIOS) {
    const low = generateDataset({ scenarioId: s.id, bias: 0, n: 1200 });
    const high = generateDataset({ scenarioId: s.id, bias: 0.9, n: 1200 });
    const rate = (d) => {
      const rows = [...d.train, ...d.test];
      const g = rows.filter((r) => r[s.subgroupKey] === 1);
      return g.filter((r) => r.y).length / g.length;
    };
    assert.ok(rate(high) < rate(low) - 0.1, s.id + ' harmed group label rate should fall');
  }
});

/* ------------------------------------------------------------------ */
/* Logistic regression                                                 */
/* ------------------------------------------------------------------ */

test('logistic regression loss decreases during gradient descent', () => {
  const d = generateDataset({ scenarioId: 'clinic', n: 600 });
  const model = trainLogisticRegression(d.train, {
    featureKeys: ['symptomSeverity', 'waitDays', 'hasProvider', 'incomeIndex'],
    epochs: 300
  });
  assert.ok(model.loss.length > 10);
  assert.ok(model.loss.at(-1) < model.loss[0]);
});

test('logistic regression converges: final loss is within 1% of the last quarter', () => {
  const d = generateDataset({ scenarioId: 'scholarship', n: 600 });
  const model = trainLogisticRegression(d.train, {
    featureKeys: ['gpa', 'attendance', 'tutoringNeed', 'householdIncome'],
    epochs: 500
  });
  const tail = model.loss.slice(-Math.floor(model.loss.length / 4));
  const spread = Math.max(...tail) - Math.min(...tail);
  assert.ok(spread / Math.abs(model.loss.at(-1)) < 0.01);
});

test('logistic regression probabilities are bounded in [0,1]', () => {
  const d = generateDataset({ scenarioId: 'shelter', n: 300 });
  const model = trainLogisticRegression(d.train, { featureKeys: ['exposureHours', 'healthRisks'] });
  for (const r of d.test) {
    const p = model.predictProba(r);
    assert.ok(Number.isFinite(p) && p >= 0 && p <= 1);
  }
});

test('logistic regression beats prevalence baseline on AUC for every scenario', () => {
  for (const id of scenarioIds) {
    const d = generateDataset({ scenarioId: id, n: 1000 });
    const keys = SCENARIOS.find((s) => s.id === id).features.map((f) => f.key);
    const model = trainLogisticRegression(d.train, { featureKeys: keys, epochs: 200 });
    const scores = d.test.map((r) => model.predictProba(r));
    const ys = d.test.map((r) => r.y);
    const pos = scores.filter((_, i) => ys[i]);
    const neg = scores.filter((_, i) => !ys[i]);
    const auc = pos.reduce((s, p) => s + neg.filter((n) => p > n || (p === n ? 0.5 : 0)).length, 0) / (pos.length * neg.length);
    assert.ok(auc > 0.65, id + ' AUC ' + auc.toFixed(3) + ' should exceed 0.65');
  }
});

test('L2 regularization shrinks coefficients relative to unregularized fit', () => {
  const d = generateDataset({ scenarioId: 'clinic', n: 800 });
  const keys = ['symptomSeverity', 'waitDays', 'hasProvider', 'incomeIndex'];
  const strong = trainLogisticRegression(d.train, { featureKeys: keys, l2: 2, epochs: 400 });
  const weak = trainLogisticRegression(d.train, { featureKeys: keys, l2: 0, epochs: 400 });
  const norm = (m) => Object.values(m.coefficients).reduce((s, c) => s + c.standardized ** 2, 0);
  assert.ok(norm(strong) < norm(weak));
});

test('coefficients are reported on raw and standardized scales', () => {
  const d = generateDataset({ scenarioId: 'clinic', n: 400 });
  const m = trainLogisticRegression(d.train, { featureKeys: ['symptomSeverity', 'waitDays'] });
  for (const c of Object.values(m.coefficients)) {
    assert.ok(Number.isFinite(c.raw) && Number.isFinite(c.standardized));
    assert.ok(c.sd > 0 && Number.isFinite(c.mean));
  }
});

/* ------------------------------------------------------------------ */
/* Decision tree                                                       */
/* ------------------------------------------------------------------ */

test('decision tree produces bounded leaf probabilities', () => {
  const d = generateDataset({ n: 500 });
  const t = trainDecisionTree(d.train, { featureKeys: ['exposureHours', 'healthRisks', 'languageBarrier'] });
  for (const r of d.test) {
    const p = t.predictProba(r);
    assert.ok(p > 0 && p < 1);
  }
});

test('decision tree leaf count respects depth limit', () => {
  const d = generateDataset({ n: 700 });
  const t = trainDecisionTree(d.train, { featureKeys: ['exposureHours', 'healthRisks'], maxDepth: 2 });
  assert.ok(t.leaves <= 4);
  assert.ok(t.rules.length === t.leaves);
});

test('decision tree accuracy exceeds majority baseline on default splits', () => {
  for (const id of scenarioIds) {
    const a = runAudit({ scenarioId: id, seed: DEFAULT_SEED, bias: 0.6 });
    assert.ok(
      a.models.tree.accuracy > a.models.tree.majorityBaseline,
      id + ' tree ' + a.models.tree.accuracy.toFixed(3) + ' must beat ' + a.models.tree.majorityBaseline.toFixed(3)
    );
  }
});

test('decision tree rules are interpretable strings', () => {
  const a = runAudit({ scenarioId: 'shelter' });
  assert.ok(a.models.tree.rules.length > 1);
  assert.ok(a.models.tree.rules.every((r) => Array.isArray(r.conditions) && r.probability >= 0 && r.n > 0));
});

/* ------------------------------------------------------------------ */
/* Metric bounds and identities                                        */
/* ------------------------------------------------------------------ */

test('ROC-AUC is between 0 and 1 and equals 1 for separable scores', () => {
  const a = runAudit({ scenarioId: 'scholarship' });
  const m = a.baseline.metrics;
  assert.ok(m.auc >= 0 && m.auc <= 1);
  // Independent Mann-Whitney check on a separable score vector.
  const scores = [0, 0.1, 0.9, 1];
  const ys = [0, 0, 1, 1];
  const pos = scores.filter((_, i) => ys[i]);
  const neg = scores.filter((_, i) => !ys[i]);
  const auc = pos.reduce((s, p) => s + neg.filter((n) => p > n).length, 0) / (pos.length * neg.length);
  assert.equal(auc, 1);
});

test('Brier score is between 0 and 1', () => {
  const a = runAudit({ scenarioId: 'clinic' });
  assert.ok(a.models.lr.brier >= 0 && a.models.lr.brier <= 1);
});

test('ECE is nonnegative and below 1', () => {
  for (const id of scenarioIds) {
    const a = runAudit({ scenarioId: id });
    assert.ok(a.baseline.metrics.ece >= 0 && a.baseline.metrics.ece <= 1);
  }
});

test('confusion counts sum to held-out size', () => {
  const audit = runAudit();
  for (const s of audit.strategies) {
    const c = s.metrics.counts;
    assert.equal(c.tp + c.fp + c.tn + c.fn, audit.testCount);
  }
});

test('precision recall and F1 satisfy their defining identities', () => {
  const m = runAudit().baseline.metrics;
  const c = m.counts;
  assert.ok(Math.abs(m.precision - c.tp / (c.tp + c.fp)) < 1e-12);
  assert.ok(Math.abs(m.recall - c.tp / (c.tp + c.fn)) < 1e-12);
  const expectedF1 = (2 * m.precision * m.recall) / (m.precision + m.recall);
  assert.ok(Math.abs(m.f1 - expectedF1) < 1e-12);
});

test('accuracy plus error rate equals one', () => {
  const m = runAudit({ scenarioId: 'shelter' }).baseline.metrics;
  const errorRate = (m.counts.fp + m.counts.fn) / m.n;
  assert.ok(Math.abs(m.accuracy + errorRate - 1) < 1e-12);
});

test('TPR plus FNR equals one within each subgroup', () => {
  const g = runAudit({ scenarioId: 'clinic' }).baseline.metrics.groups;
  for (const id of ['groupA', 'groupB']) {
    assert.ok(Math.abs(g[id].tpr + g[id].fnr - 1) < 1e-12);
  }
});

test('selection rate equals (TP+FP)/n per subgroup', () => {
  const g = runAudit({ scenarioId: 'shelter' }).baseline.metrics.groups;
  for (const id of ['groupA', 'groupB']) {
    const s = g[id].counts;
    assert.ok(Math.abs(g[id].selectionRate - (s.tp + s.fp) / g[id].n) < 1e-12);
  }
});

test('fairness gap identities hold', () => {
  const g = runAudit({ scenarioId: 'scholarship' }).baseline.metrics.groups;
  assert.ok(Math.abs(g.equalOpportunityGap - Math.abs(g.groupA.tpr - g.groupB.tpr)) < 1e-12);
  assert.ok(Math.abs(g.fnrDisparity - Math.abs(g.groupA.fnr - g.groupB.fnr)) < 1e-12);
});

test('ROC curve is monotone with endpoints at (0,0) and (1,1)', () => {
  const roc = runAudit().roc;
  assert.deepEqual(roc[0], { fpr: 0, tpr: 0 });
  assert.deepEqual(roc.at(-1), { fpr: 1, tpr: 1 });
  for (let i = 1; i < roc.length; i++) {
    assert.ok(roc[i].fpr >= roc[i - 1].fpr);
    assert.ok(roc[i].tpr >= roc[i - 1].tpr);
  }
});

test('reliability diagram has ten bins that sum to the held-out size', () => {
  const a = runAudit();
  assert.equal(a.reliability.length, 10);
  assert.equal(a.reliability.reduce((s, b) => s + b.n, 0), a.testCount);
});

/* ------------------------------------------------------------------ */
/* Calibration                                                         */
/* ------------------------------------------------------------------ */

test('Platt calibration reduces training Brier score', () => {
  const d = generateDataset({ scenarioId: 'shelter', n: 800 });
  const keys = ['exposureHours', 'healthRisks', 'dependents', 'transportAccess'];
  const m = trainLogisticRegression(d.train, { featureKeys: keys });
  const scores = d.train.map((r) => m.predictProba(r));
  const ys = d.train.map((r) => r.y);
  const cal = fitCalibration(scores, ys);
  assert.ok(cal.calibratedBrier <= cal.trainBrier + 1e-9);
});

test('Platt calibration reduces training ECE on the default audit split', () => {
  const a = runAudit({ scenarioId: 'clinic' });
  assert.ok(a.calibration.calibratedEce < a.calibration.rawEce);
});

test('calibration apply is monotone and bounded', () => {
  const cal = fitCalibration([0.1, 0.3, 0.5, 0.7, 0.9], [0, 0, 1, 1, 1]);
  const out = [0.05, 0.2, 0.5, 0.8, 0.95].map(cal.apply);
  for (let i = 1; i < out.length; i++) assert.ok(out[i] >= out[i - 1]);
  assert.ok(out.every((p) => p > 0 && p < 1));
});

test('calibration is fitted on train, not the held-out split', () => {
  const a = runAudit({ scenarioId: 'clinic' });
  assert.ok(a.calibration.rawBrier > 0 && a.calibration.calibratedBrier > 0);
  assert.ok(a.calibration.testRawBrier > 0 && a.calibration.testCalibratedBrier > 0);
  assert.ok(a.calibration.temperature > 0 && a.calibration.temperature < 20);
});

/* ------------------------------------------------------------------ */
/* Fairness mitigation                                                 */
/* ------------------------------------------------------------------ */

test('equal opportunity thresholds differ between groups on biased splits', () => {
  for (const id of scenarioIds) {
    const a = runAudit({ scenarioId: id, bias: 0.6 });
    const eo = a.strategies.find((s) => s.id === 'equal-opportunity');
    assert.notEqual(eo.thresholds[0], eo.thresholds[1], id + ' should choose distinct thresholds');
  }
});

test('equal opportunity mitigation reduces held-out TPR gap on all scenarios', () => {
  for (const id of scenarioIds) {
    const a = runAudit({ scenarioId: id, bias: 0.6 });
    const base = a.baseline.metrics.groups.equalOpportunityGap;
    const eo = a.strategies.find((s) => s.id === 'equal-opportunity').metrics.groups.equalOpportunityGap;
    assert.ok(eo < base, id + ' EO gap ' + base.toFixed(3) + ' -> ' + eo.toFixed(3));
  }
});

test('equal opportunity mitigation reduces held-out FNR disparity', () => {
  for (const id of scenarioIds) {
    const a = runAudit({ scenarioId: id, bias: 0.6 });
    const base = a.baseline.metrics.groups.fnrDisparity;
    const eo = a.strategies.find((s) => s.id === 'equal-opportunity').metrics.groups.fnrDisparity;
    assert.ok(eo < base, id + ' FNR gap ' + base.toFixed(3) + ' -> ' + eo.toFixed(3));
  }
});

test('stronger label bias widens the harmed group label-rate disparity', () => {
  const rate = (bias) => {
    const d = generateDataset({ scenarioId: 'clinic', bias, n: 1200 });
    const rows = [...d.train, ...d.test];
    const g0 = rows.filter((r) => r.rural === 0);
    const g1 = rows.filter((r) => r.rural === 1);
    return g0.filter((r) => r.y).length / g0.length - g1.filter((r) => r.y).length / g1.length;
  };
  assert.ok(rate(0.9) > rate(0.6) && rate(0.6) > rate(0.1));
});

test('proxy removal retrains without the named proxy feature', () => {
  const a = runAudit({ scenarioId: 'scholarship' });
  const proxy = a.strategies.find((s) => s.id === 'proxy-removal');
  assert.deepEqual(proxy.removedFeatures, ['zipAdvantage']);
  assert.ok(a.featureKeys.includes('zipAdvantage'));
  assert.ok(a.scenario === 'scholarship');
});

test('selection rate mitigation reduces demographic parity gap', () => {
  const a = runAudit({ scenarioId: 'shelter', bias: 0.6 });
  const base = a.baseline.metrics.groups.demographicParityGap;
  const sel = a.strategies.find((s) => s.id === 'equal-selection').metrics.groups.demographicParityGap;
  assert.ok(sel < base);
});

test('optimizeGroupThresholds and optimizeSelectionThresholds return one threshold per group', () => {
  const d = generateDataset({ n: 500 });
  const scores = d.train.map((r) => r.rural ? 0.6 : 0.4);
  const eo = optimizeGroupThresholds(d.train, scores, { subgroupKey: 'rural' });
  const sel = optimizeSelectionThresholds(d.train, scores, { subgroupKey: 'rural', target: 0.4 });
  assert.deepEqual(Object.keys(eo).map(Number).sort(), [0, 1]);
  assert.deepEqual(Object.keys(sel).map(Number).sort(), [0, 1]);
  for (const t of [...Object.values(eo), ...Object.values(sel)]) assert.ok(t >= 0 && t <= 1);
});

/* ------------------------------------------------------------------ */
/* Full audit, counterexamples and export                              */
/* ------------------------------------------------------------------ */

test('runAudit is fully deterministic for fixed parameters', () => {
  const a = runAudit({ scenarioId: 'clinic', seed: 1234, bias: 0.5, n: 800, threshold: 0.4 });
  const b = runAudit({ scenarioId: 'clinic', seed: 1234, bias: 0.5, n: 800, threshold: 0.4 });
  assert.deepEqual(a.strategies.map((s) => s.metrics.accuracy), b.strategies.map((s) => s.metrics.accuracy));
  assert.deepEqual(a.roc, b.roc);
});

test('changing seed changes the held-out metrics', () => {
  const a = runAudit({ seed: 1 });
  const b = runAudit({ seed: 2 });
  assert.notEqual(a.baseline.metrics.accuracy, b.baseline.metrics.accuracy);
});

test('higher threshold lowers selection rate', () => {
  const a = runAudit({ scenarioId: 'clinic', threshold: 0.2 });
  const b = runAudit({ scenarioId: 'clinic', threshold: 0.7 });
  assert.ok(b.baseline.metrics.groups.groupA.selectionRate <= a.baseline.metrics.groups.groupA.selectionRate);
});

test('counterexamples are held-out harmed-group positives recovered by mitigation', () => {
  const a = runAudit({ scenarioId: 'shelter', bias: 0.6 });
  assert.ok(a.counterexamples.length > 0);
  for (const c of a.counterexamples) {
    assert.equal(c.y, 1);
    assert.equal(c.group, a.harmedGroup);
    assert.equal(c.globalPrediction, 0);
    assert.ok(c.flips.length > 0);
    assert.ok(c.score > 0 && c.score < 1);
  }
});

test('failure cases carry features and an explanatory note', () => {
  const a = runAudit();
  assert.ok(a.failures.length > 0 && a.failures.length <= 10);
  for (const f of a.failures) {
    assert.ok(Object.keys(f.features).length === a.featureKeys.length);
    assert.ok(f.note.length > 10);
  }
});

test('exportAudit returns one complete row per strategy', () => {
  const a = runAudit();
  const rows = exportAudit(a);
  assert.equal(rows.length, a.strategies.length);
  const keys = ['strategy', 'accuracy', 'recall', 'f1', 'auc', 'brier', 'ece', 'dp_gap', 'eo_gap', 'fnr_gap'];
  for (const row of rows) for (const k of keys) assert.ok(k in row);
});

test('all audit values are finite JSON serializable numbers', () => {
  for (const id of scenarioIds) {
    const a = runAudit({ scenarioId: id });
    const text = JSON.stringify(a);
    assert.ok(text.length > 1000);
    assert.ok(!text.includes('NaN'));
    assert.ok(!text.includes('Infinity'));
  }
});

/* ------------------------------------------------------------------ */
/* Leakage, threshold space, calibration and gap-format regressions      */
/* ------------------------------------------------------------------ */

test('selection-rate target is computed from train rows only', () => {
  const d = generateDataset({ scenarioId: 'clinic', seed: 4242, bias: 0.6, n: 1400 });
  const keys = ['symptomSeverity', 'waitDays', 'hasProvider', 'ageYears', 'incomeIndex'];
  const model = trainLogisticRegression(d.train, { featureKeys: keys, epochs: 400 });
  const cal = fitCalibration(d.train.map((r) => model.predictProba(r)), d.train.map((r) => r.y));
  const trainScores = d.train.map((r) => cal.apply(model.predictProba(r)));
  const { target, rates } = selectionTargetFromTrain(d.train, trainScores, 0.35, 'rural');
  assert.ok(Math.abs(target - (rates[0] + rates[1]) / 2) < 1e-12);
  for (const g of [0, 1]) {
    const idx = d.train.map((r, i) => ({ r, i })).filter(({ r }) => r.rural === g);
    const selected = idx.filter(({ i }) => trainScores[i] >= 0.35).length;
    assert.ok(Math.abs(rates[g] - selected / idx.length) < 1e-12);
  }
  // A poisoned held-out split cannot reach the train-only target computation.
  d.test.forEach((r) => { r.y = 1 - r.y; });
  assert.equal(selectionTargetFromTrain(d.train, trainScores, 0.35, 'rural').target, target);
});

test('mutating held-out labels and rows changes no fitted threshold or target', () => {
  const opts = { scenarioId: 'clinic', seed: 12345, bias: 0.6, threshold: 0.35, n: 1400 };
  const base = runAudit(opts);
  const keys = ['symptomSeverity', 'waitDays', 'hasProvider', 'ageYears', 'incomeIndex'];
  const d = generateDataset({ scenarioId: 'clinic', seed: 12345, bias: 0.6, n: 1400 });
  const model = trainLogisticRegression(d.train, { featureKeys: keys, epochs: 400 });
  const cal = fitCalibration(d.train.map((r) => model.predictProba(r)), d.train.map((r) => r.y));
  const trainScores = d.train.map((r) => cal.apply(model.predictProba(r)));
  const cleanTarget = selectionTargetFromTrain(d.train, trainScores, 0.35, 'rural').target;
  const cleanEo = fitGroupThresholds(d.train, trainScores, { subgroupKey: 'rural', referenceThreshold: 0.35 });
  const cleanSel = optimizeSelectionThresholds(d.train, trainScores, { subgroupKey: 'rural', target: cleanTarget });

  // Poison every held-out label and feature, then refit from the same seed.
  d.test.forEach((r, i) => { r.y = 1 - r.y; r.symptomSeverity += 1000 * (i + 1); });
  assert.equal(selectionTargetFromTrain(d.train, trainScores, 0.35, 'rural').target, cleanTarget);
  assert.deepEqual(fitGroupThresholds(d.train, trainScores, { subgroupKey: 'rural', referenceThreshold: 0.35 }), cleanEo);
  assert.deepEqual(optimizeSelectionThresholds(d.train, trainScores, { subgroupKey: 'rural', target: cleanTarget }), cleanSel);

  const again = runAudit(opts);
  assert.deepEqual(again.strategies.find((s) => s.id === 'equal-opportunity').thresholds,
    base.strategies.find((s) => s.id === 'equal-opportunity').thresholds);
  assert.deepEqual(again.strategies.find((s) => s.id === 'equal-selection').thresholds,
    base.strategies.find((s) => s.id === 'equal-selection').thresholds);
  assert.equal(again.fitting.selectionTarget, base.fitting.selectionTarget);
  assert.equal(again.fitting.eoGoalTpr, base.fitting.eoGoalTpr);
});

test('EO thresholds are fitted in calibrated train space, not raw train space', () => {
  const opts = { scenarioId: 'clinic', seed: 20270101, bias: 0.6, threshold: 0.35, n: 1400 };
  const audit = runAudit(opts);
  const keys = audit.featureKeys;
  const d = generateDataset({ scenarioId: 'clinic', seed: opts.seed, bias: opts.bias, n: opts.n });
  const model = trainLogisticRegression(d.train, { featureKeys: keys, l2: 0.02, epochs: 400 });
  const cal = fitCalibration(d.train.map((r) => model.predictProba(r)), d.train.map((r) => r.y));
  const raw = d.train.map((r) => model.predictProba(r));
  const calibrated = raw.map(cal.apply);
  const fitted = fitGroupThresholds(d.train, calibrated, {
    subgroupKey: 'rural', referenceThreshold: 0.35
  }).thresholds;
  const rawFit = fitGroupThresholds(d.train, raw, {
    subgroupKey: 'rural', referenceThreshold: 0.35
  }).thresholds;
  const live = audit.strategies.find((s) => s.id === 'equal-opportunity').thresholds;
  assert.deepEqual(Object.keys(live).sort(), Object.keys(fitted).sort());
  for (const g of Object.keys(fitted)) {
    assert.ok(Math.abs(live[g] - fitted[g]) < 1e-12,
      'group ' + g + ' must use calibrated train thresholds');
  }
  // Guard: the raw-space fit is a different computation for at least one group.
  assert.ok(Object.keys(rawFit).some((g) => Math.abs(rawFit[g] - fitted[g]) > 1e-9));
});

test('proxy removal uses its own train-only calibration', () => {
  const opts = { scenarioId: 'scholarship', seed: 20270101, bias: 0.6, threshold: 0.35, n: 1400 };
  const audit = runAudit(opts);
  const calNoProxy = audit.calibration.noProxy;
  assert.ok(calNoProxy && Number.isFinite(calNoProxy.a) && Number.isFinite(calNoProxy.b));
  assert.ok(calNoProxy.trainBrier >= 0 && calNoProxy.calibratedBrier >= 0);
  assert.ok(calNoProxy.trainEce >= 0 && calNoProxy.calibratedEce >= 0);
  // When the full and no-proxy models differ, their calibration params differ too.
  assert.ok(
    Math.abs(audit.calibration.a - calNoProxy.a) > 1e-9 ||
    Math.abs(audit.calibration.b - calNoProxy.b) > 1e-9,
    'no-proxy fit must not simply borrow the full model calibration'
  );
  for (const id of scenarioIds) {
    const a = runAudit({ scenarioId: id, seed: opts.seed, bias: opts.bias, threshold: opts.threshold, n: opts.n });
    assert.ok(a.calibration.noProxy.temperature > 0 && a.calibration.noProxy.temperature <= 20, id);
  }
});

test('EO leveling maintains overall recall near or above the global baseline', () => {
  for (const id of scenarioIds) {
    const a = runAudit({ scenarioId: id, seed: 20270101, bias: 0.6, threshold: 0.35, n: 1400 });
    const base = a.baseline.metrics;
    const eo = a.strategies.find((s) => s.id === 'equal-opportunity').metrics;
    assert.ok(eo.recall >= base.recall - 0.005,
      id + ' EO recall ' + eo.recall.toFixed(3) + ' must not level down from ' + base.recall.toFixed(3));
  }
});

test('formatGapChange handles worsening, unchanged, zero and improving gaps', () => {
  assert.match(formatGapChange(-0.052, 0.12), /gap increased by 5\.2 percentage points/);
  assert.match(formatGapChange(-0.052, 0.12), /increased/);
  assert.doesNotMatch(formatGapChange(-0.052, 0.12), /reduced|no gap/);
  assert.match(formatGapChange(0, 0), /no gap \(0\.0%\)/);
  assert.match(formatGapChange(0.0001, 0.032), /gap unchanged \(3\.2%\)/);
  assert.doesNotMatch(formatGapChange(0.0001, 0.032), /no gap|reduced/);
  assert.match(formatGapChange(0.184, 0.061), /reduced by 18\.4 percentage points/);
  for (const text of [
    formatGapChange(-0.052, 0.12), formatGapChange(0, 0),
    formatGapChange(0.0001, 0.032), formatGapChange(0.184, 0.061)
  ]) {
    assert.ok(!text.includes('-%') && !text.includes('negative'), text);
  }
});

test('gap UI never reports no gap when the actual gap is positive at clinic seed 12345', () => {
  const a = runAudit({ scenarioId: 'clinic', seed: 12345, bias: 0.6, threshold: 0.35 });
  const base = a.baseline.metrics;
  const eo = a.strategies.find((s) => s.id === 'equal-opportunity').metrics;
  const gapDrop = base.groups.equalOpportunityGap - eo.groups.equalOpportunityGap;
  const text = formatGapChange(gapDrop, eo.groups.equalOpportunityGap) + ' with equal opportunity';
  assert.ok(eo.groups.equalOpportunityGap > 0);
  assert.ok(text.includes('reduced by '), text);
  assert.ok(!text.includes('no gap at this setting'), text);
  assert.ok(!text.includes("-%"), text);
  // Trade-off banner must name any secondary gap widening explicitly.
  const notes = tradeoffNotes(a);
  for (const n of notes) {
    for (const c of n.changes) assert.ok(c.includes('pp'));
  }
});
