/* Wave-5 adversarial regression tests. Each test guards a defect the
   hostile audit reproduced against the wave-4 suite. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import {
  generateDataset,
  runAudit,
  parseCsv,
  validateCsvRows,
  subgroupMetrics,
  fairnessSummary,
  fitCalibration,
  optimizeSelectionThresholds,
  trainDecisionTree,
  exportAudit,
  toCsv,
  DATA_STATEMENT
} from '../src/ml-engine.js';

const here = fileURLToPath(new URL('.', import.meta.url));
const appSrc = readFileSync(join(here, '../src/app.js'), 'utf8');
const indexSrc = readFileSync(join(here, '../index.html'), 'utf8');

/* ---- M1: fractional subgroup values must stay distinct groups ---- */

test('fractional subgroup keys form distinct groups instead of collapsing to 0', () => {
  const rows = [
    { g: 0.5, y: 1 }, { g: 0.5, y: 1 },
    { g: 0.9, y: 1 }, { g: 0.9, y: 1 }
  ];
  const scores = [0.9, 0.9, 0.1, 0.1];
  const groups = subgroupMetrics(rows, scores, 0.5, 'g');
  const keys = Object.keys(groups).map(Number).sort((a, b) => a - b);
  assert.deepEqual(keys, [0.5, 0.9]);
  const summary = fairnessSummary(groups, 'g');
  assert.ok(summary.equalOpportunityGap > 0.9, 'real TPR gap must not report as zero');
});

test('negative fractional and large keys do not truncate or wrap', () => {
  const rows = [{ g: -0.4, y: 1 }, { g: 0.4, y: 1 }, { g: 2147483648, y: 0 }];
  const groups = subgroupMetrics(rows, [0.9, 0.1, 0.2], 0.5, 'g');
  assert.equal(Object.keys(groups).length, 3);
  assert.ok(groups['-0.4'] && groups['0.4'] && groups['2147483648']);
});

/* ---- M2: string subgroup ids are order-invariant across arrays ---- */

test('string subgroup labels map to the same group id regardless of row order', () => {
  const rural = (i) => ({ g: 'rural', y: 1 });
  const urban = (i) => ({ g: 'urban', y: i % 2 });
  const a = [...Array.from({ length: 4 }, rural), ...Array.from({ length: 4 }, urban)];
  const b = [...Array.from({ length: 4 }, urban), ...Array.from({ length: 4 }, rural)];
  const scores = Array(8).fill(0.6);
  const ga = subgroupMetrics(a, scores, 0.5, 'g');
  const gb = subgroupMetrics(b, scores, 0.5, 'g');
  assert.deepEqual(Object.keys(ga).sort(), Object.keys(gb).sort());
  // The all-positive (rural) cohort must land on the same key in both
  // orderings, so a fitted threshold can never swap groups between arrays.
  const ruralKeyA = Object.keys(ga).find((k) => ga[k].positives === 4);
  const ruralKeyB = Object.keys(gb).find((k) => gb[k].positives === 4);
  assert.equal(ruralKeyA, ruralKeyB);
});

test('fitted threshold maps are stable when the same rows arrive reordered', () => {
  const rows = [
    { g: 'rural', y: 1 }, { g: 'rural', y: 1 }, { g: 'rural', y: 0 }, { g: 'rural', y: 0 },
    { g: 'urban', y: 1 }, { g: 'urban', y: 1 }, { g: 'urban', y: 0 }, { g: 'urban', y: 0 }
  ];
  const scores = [0.9, 0.8, 0.6, 0.4, 0.5, 0.45, 0.2, 0.1];
  const shuffledIdx = [4, 5, 6, 7, 0, 1, 2, 3];
  const t1 = optimizeSelectionThresholds(rows, scores, { subgroupKey: 'g', target: 0.5 });
  const t2 = optimizeSelectionThresholds(
    shuffledIdx.map((i) => rows[i]),
    shuffledIdx.map((i) => scores[i]),
    { subgroupKey: 'g', target: 0.5 }
  );
  assert.deepEqual(t1, t2);
});

/* ---- m1: proxy-removal model honors the same training budget ---- */

test('proxy-removal strategy responds to the epochs option', () => {
  const a5 = runAudit({ epochs: 5, seed: 11 });
  const a400 = runAudit({ epochs: 400, seed: 11 });
  const p5 = a5.strategies.find((s) => s.id === 'proxy-removal');
  const p400 = a400.strategies.find((s) => s.id === 'proxy-removal');
  assert.ok(p5 && p400);
  assert.notDeepEqual(
    { acc: p5.metrics.accuracy, thr: p5.thresholds ?? p5.threshold },
    { acc: p400.metrics.accuracy, thr: p400.thresholds ?? p400.threshold },
    'proxy-removal must train with the same epochs option as the main model'
  );
});

/* ---- m2: bias metadata must match the applied bias ---- */

test('generateDataset applies the clamped bias the metadata records', () => {
  const d0 = generateDataset({ seed: 7, bias: 0, n: 2000 });
  const d9 = generateDataset({ seed: 7, bias: '0.9', n: 2000 });
  assert.equal(d9.bias, 0.9);
  const rate = (d) =>
    d.train.concat(d.test).filter((r) => r.y === 1).length / (d.train.length + d.test.length);
  assert.ok(rate(d9) < rate(d0), 'a recorded 0.9 bias must actually be applied');
});

/* ---- m3: calibration may soften overconfident scores ---- */

test('fitCalibration can return a slope below 1 (softening)', () => {
  // Correctly ranked but wildly overconfident scores: the only honest
  // calibration pulls them toward the observed rate.
  const scores = [0.95, 0.92, 0.9, 0.88, 0.85, 0.82, 0.9, 0.94, 0.87, 0.91];
  const ys =     [1,    0,    0,   1,    0,    1,   0,    0,    1,    0];
  const cal = fitCalibration(scores, ys);
  assert.ok(cal.a <= 1, 'slope ' + cal.a + ' should allow softening');
  assert.ok(cal.calibratedBrier <= cal.trainBrier + 1e-9);
});

/* ---- m4: the CSV export carries the synthetic-data statement ---- */

test('the CSV download prepends the synthetic-data statement', () => {
  assert.match(appSrc, /DATA_STATEMENT/);
  assert.match(appSrc, /#\s*'\s*\+\s*DATA_STATEMENT/);
  assert.match(DATA_STATEMENT, /synthetic/i);
  const audit = runAudit({ seed: 3, n: 200 });
  const csv = '# ' + DATA_STATEMENT.replace(/\s+/g, ' ') + '\r\n' + toCsv(exportAudit(audit));
  assert.ok(csv.startsWith('# ' + DATA_STATEMENT.slice(0, 40)));
});

/* ---- m5: suffixed header collisions keep every column ---- */

test('duplicate headers with a colliding suffix keep all columns', () => {
  const parsed = parseCsv('a,a__2,a\n1,2,3');
  assert.deepEqual(parsed.headers, ['a', 'a__2', 'a__3']);
  assert.deepEqual([parsed.rows[0].a, parsed.rows[0].a__2, parsed.rows[0].a__3], [1, 2, 3]);
  assert.ok(parsed.duplicates.includes('a'));
});

/* ---- m6: an empty parse is invalid, not vacuously valid ---- */

test('validateCsvRows fails closed on zero rows', () => {
  assert.equal(validateCsvRows([]).valid, false);
  assert.equal(validateCsvRows(parseCsv('').rows).valid, false);
  assert.equal(validateCsvRows(parseCsv('h1,h2\n').rows).valid, false);
  assert.equal(validateCsvRows([{ a: 1 }]).valid, true);
});

/* ---- m7: subgroupMetrics bounds inputs and defaults the threshold ---- */

test('subgroupMetrics ignores rows with no paired score and defaults NaN threshold', () => {
  const rows = [{ g: 0, y: 1 }, { g: 1, y: 1 }, { g: 0, y: 0 }, { g: 1, y: 0 }, { g: 0, y: 1 }, { g: 1, y: 1 }];
  const bounded = subgroupMetrics(rows, [0.9, 0.1, 0.2], 0.5, 'g');
  const total = Object.values(bounded).reduce((s, g) => s + g.n, 0);
  assert.equal(total, 3);
  const g = subgroupMetrics([{ g: 0, y: 1 }, { g: 1, y: 0 }], [0.6, 0.4], NaN, 'g');
  assert.equal(g[0].selectionRate, 1, 'score 0.6 selects at the 0.5 default');
  assert.equal(g[1].selectionRate, 0, 'score 0.4 does not select at the 0.5 default');
});

/* ---- m8: the tree treats every missing representation consistently ---- */

test('tree predict routes null, undefined and non-numeric the same way', () => {
  const rows = Array.from({ length: 200 }, (_, i) => ({ f: i / 200, y: i > 99 ? 1 : 0 }));
  const t = trainDecisionTree(rows, { featureKeys: ['f'], minLeaf: 8 });
  const pMissing = t.predictProba({});
  assert.equal(t.predictProba({ f: null }), pMissing);
  assert.equal(t.predictProba({ f: undefined }), pMissing);
  assert.equal(t.predictProba({ f: 'abc' }), pMissing);
  assert.equal(t.predictProba(null), pMissing);
  assert.ok(Number.isFinite(pMissing));
});

/* ---- m9: reported tree depth is the achieved depth, not the cap ---- */

test('tree.depth reports the achieved depth', () => {
  const rows = Array.from({ length: 200 }, (_, i) => ({ f: i, y: i > 99 ? 1 : 0 }));
  const t = trainDecisionTree(rows, { featureKeys: ['f'], maxDepth: 5, minLeaf: 8 });
  assert.equal(t.maxDepth, 5);
  assert.equal(t.depth, 1);
  const audit = runAudit({ seed: 5, n: 400 });
  assert.ok(audit.models.tree.depth <= audit.models.tree.rules.length);
  assert.ok(audit.models.tree.depth <= 3);
});

/* ---- m10: aria-live is consolidated; case cards have concise names ---- */

test('live regions are consolidated instead of announcing five at once', () => {
  const liveCount = (indexSrc.match(/aria-live=/g) || []).length;
  assert.ok(liveCount <= 2, 'expected at most the banner and toast live regions');
});

test('counterexample cards expose a concise accessible name', () => {
  assert.match(appSrc, /setAttribute\('aria-label',/);
  assert.match(appSrc, /Held-out case ' \+ c\.id/);
});
