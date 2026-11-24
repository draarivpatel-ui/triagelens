/* Wave 9 revamp regressions: fairness arithmetic, hostile hyperparameters,
 * malformed CSV, deterministic ordering and keyboard-accessible chart names. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  generateDataset,
  trainLogisticRegression,
  trainDecisionTree,
  fitCalibration,
  parseCsv,
  fairnessSummary,
  classMetrics
} from '../src/ml-engine.js';
import {
  fitSensitivityFloor,
  fitRateMatchThresholds,
  sliceMetrics,
  thresholdFrontier
} from '../src/calibration.js';
import { drawReliabilitySvg, drawFrontierSvg } from '../src/svg-charts.js';

const metric = (over = {}) => ({
  n: 10, positives: 5, prevalence: 0.5, selectionRate: 0.5,
  tpr: 0.5, fnr: 0.5, fpr: 0.5, ppv: 0.5, npv: 0.5, calibrationError: 0,
  counts: { tp: 2, fp: 2, tn: 3, fn: 3 }, ...over
});

test('demographic parity gap is the absolute selection-rate difference', () => {
  const f = fairnessSummary({
    0: metric({ selectionRate: 0.4 }),
    1: metric({ selectionRate: 0.2 })
  }, 'g');
  assert.equal(f.demographicParityGap, 0.2);
});

test('fairnessSummary survives non-numeric group keys deterministically', () => {
  const f = fairnessSummary({
    urban: metric({ selectionRate: 0.2, tpr: 0.4 }),
    rural: metric({ selectionRate: 0.7, tpr: 0.9 })
  }, 'g');
  assert.ok(Math.abs(f.demographicParityGap - 0.5) < 1e-12);
  assert.ok(Math.abs(f.equalOpportunityGap - 0.5) < 1e-12);
  assert.deepEqual(f.groups, ['rural', 'urban']);
});

test('classMetrics reports undefined degenerate error rates as zero', () => {
  const noPositives = classMetrics({ tp: 0, fp: 2, tn: 8, fn: 0 });
  assert.equal(noPositives.recall, 0);
  assert.equal(noPositives.fnr, 0);
  const noNegatives = classMetrics({ tp: 4, fp: 0, tn: 0, fn: 1 });
  assert.equal(noNegatives.specificity, 0);
  assert.equal(noNegatives.fpr, 0);
});

test('logistic training normalizes hostile keys and hyperparameters', () => {
  const rows = generateDataset({ n: 120, seed: 17 }).train;
  const model = trainLogisticRegression(rows, {
    featureKeys: 'symptomSeverity',
    l2: NaN,
    lr: NaN,
    epochs: NaN
  });
  assert.ok(Array.isArray(model.keys));
  for (const coef of Object.values(model.coefficients)) {
    assert.ok(Number.isFinite(coef.raw) && Number.isFinite(coef.standardized));
  }
  const p = model.predictProba(rows[0]);
  assert.ok(Number.isFinite(p) && p >= 0 && p <= 1, 'p=' + p);
});

test('decision tree normalizes hostile keys and depth options', () => {
  const rows = generateDataset({ n: 120, seed: 17 }).train;
  const tree = trainDecisionTree(rows, {
    featureKeys: 'symptomSeverity',
    maxDepth: NaN,
    minLeaf: NaN
  });
  assert.ok(Array.isArray(tree.keys));
  const p = tree.predictProba(rows[0]);
  assert.ok(Number.isFinite(p) && p >= 0 && p <= 1, 'p=' + p);
});

test('calibration never leaks non-finite parameters or losses', () => {
  const cal = fitCalibration([0.2, 0.8], [0, 1], {
    epochs: NaN,
    lr: NaN,
    initA: NaN,
    initB: NaN
  });
  for (const v of [cal.a, cal.b, cal.temperature, cal.loss, ...cal.calibrate]) {
    assert.ok(Number.isFinite(v), 'non-finite calibration value ' + v);
  }
  assert.equal(cal.apply('0.8'), cal.apply(0.8));
  assert.ok(cal.apply(null) >= 0 && cal.apply(null) <= 1);
});

test('parseCsv rejects an unterminated quoted row even when its width matches', () => {
  const p = parseCsv('a\n"unterminated\n');
  assert.equal(p.rows.length, 0);
  assert.equal(p.skipped, 1);
});

test('sensitivity floor is clamped to the unit interval', () => {
  const fit = fitSensitivityFloor([0.2, 0.8], [1, 0], 2);
  assert.equal(fit.floor, 1);
  assert.equal(fit.achievable, true);
  assert.equal(fit.recall, 1);
  const low = fitSensitivityFloor([0.2, 0.8], [1, 0], -1);
  assert.equal(low.floor, 0);
});

test('rate-match thresholds use deterministic fractional group order', () => {
  const fixture = (groups) => {
    const rows = [];
    const scores = [];
    for (const g of groups) {
      for (let i = 0; i < 6; i++) {
        rows.push({ g, y: i < 2 ? 1 : 0 });
        scores.push(i < 2 ? 0.8 : 0.2);
      }
    }
    return fitRateMatchThresholds(rows, scores, 'g', 0.5);
  };
  const a = fixture([0.5, 0.25]);
  const b = fixture([0.25, 0.5]);
  assert.deepEqual(Object.keys(a.thresholds), ['0.25', '0.5']);
  assert.deepEqual(Object.keys(a.fitDetail), ['0.25', '0.5']);
  assert.deepEqual(a, b);
});

test('sliceMetrics orders non-numeric ids deterministically', () => {
  const input = {
    rows: [{ y: 1 }, { y: 0 }, { y: 1 }, { y: 0 }],
    scores: [0.8, 0.2, 0.7, 0.3],
    groupIds: ['b', 'a', 'a', 'b'],
    sliceIds: ['z', 'y', 'y', 'z'],
    threshold: 0.5
  };
  const sm = sliceMetrics(input);
  assert.deepEqual(sm.cells.map((c) => c.id), ['a|y', 'b|z']);
});

test('thresholdFrontier orders non-numeric group ids deterministically', () => {
  const f = thresholdFrontier([0.8, 0.2], [1, 0], ['b', 'a']);
  assert.deepEqual(f.groupIds, ['a', 'b']);
});

/* Tiny DOM stand-in for SVG keyboard-name tests. */
class StubEl {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attrs = {};
    this.listeners = {};
    this.textContent = '';
    this.style = {};
  }
  appendChild(c) { this.children.push(c); return c; }
  replaceChildren(...cs) { this.children = cs; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  addEventListener(t, fn) { this.listeners[t] = fn; }
}
globalThis.document = {
  createElementNS: (_ns, tag) => new StubEl(tag),
  documentElement: new StubEl('html')
};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });

test('SVG reliability keyboard stops expose the full bin reading', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), {
    bins: [{ bin: 5, lo: 0.5, hi: 0.6, n: 12, confidence: 0.55, observed: 0.5, observedCi: { lo: 0.2, hi: 0.8 } }]
  });
  const point = svg.children.find((c) => c.tagName === 'g' && c.attrs.tabindex === '0');
  assert.ok(point.attrs['aria-label'].includes('predicted 0.550'));
  assert.ok(point.attrs['aria-label'].includes('observed 0.500'));
  assert.ok(point.attrs['aria-label'].includes('n=12'));
});

test('SVG policy keyboard stops expose operating-point values', () => {
  const svg = drawFrontierSvg(new StubEl('div'), {
    points: [],
    policyPoints: [{ id: 'global', label: 'Global', sel: 0.4, fnr: 0.25 }]
  });
  const point = svg.children.find((c) => c.tagName === 'g' && c.attrs['data-policy'] === 'global');
  assert.ok(point.attrs['aria-label'].includes('Global'));
  assert.ok(point.attrs['aria-label'].includes('alerts 40.0'));
  assert.ok(point.attrs['aria-label'].includes('missed 25.0'));
});

test('theme toggle exposes and updates a pressed state', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const app = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
  assert.match(html, /id="theme-toggle"[^>]*aria-pressed="false"/);
  assert.match(app, /setAttribute\('aria-pressed'/);
});

test('model weight labels wrap instead of truncating on narrow cards', () => {
  const css = readFileSync(new URL('../styles.css', import.meta.url), 'utf8');
  const app = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
  const row = css.slice(css.indexOf('.weight-row {'), css.indexOf('.weight-track {'));
  assert.match(row, /grid-template-columns:\s*minmax\(0,\s*1fr\)/, 'label column is not fluid');
  assert.match(row, /grid-column:\s*1\s*\/\s*-1/, 'label is not given the full row');
  assert.doesNotMatch(row, /text-overflow:\s*ellipsis/, 'label still ellipsizes');
  assert.doesNotMatch(row, /white-space:\s*nowrap/, 'label cannot wrap');
  assert.match(app, /label\.title\s*=\s*label\.textContent/, 'full label is not available on hover');
});
