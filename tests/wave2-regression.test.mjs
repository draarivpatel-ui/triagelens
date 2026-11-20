/* Wave 2 targeted regression tests: statistical identities, CSV validation,
 * demo wiring and visual-rule invariants. No dependencies, no network. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import {
  runAudit,
  rocAuc,
  brierDecomposition,
  brierScore,
  subgroupMetrics,
  fairnessSummary,
  validateCsvRows,
  parseCsv,
  exportAudit,
  fitGroupThresholds,
  generateDataset,
  trainLogisticRegression,
  fitCalibration,
  SCENARIOS
} from '../src/ml-engine.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const read = (p) => readFileSync(resolve(root, p), 'utf8');

/* ------------------------------------------------------------------ */
/* ROC-AUC tie handling vs an independent Mann-Whitney reference        */
/* ------------------------------------------------------------------ */

function referenceAuc(scores, ys) {
  const pos = scores.filter((_, i) => ys[i]);
  const neg = scores.filter((_, i) => !ys[i]);
  if (!pos.length || !neg.length) return 0.5;
  let wins = 0;
  for (const p of pos) for (const n of neg) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}

test('ROC-AUC gives half credit for every tie (Mann-Whitney identity)', () => {
  const cases = [
    [[0.5, 0.5], [0, 1]],
    [[0.5, 0.5, 0.5], [1, 1, 0]],
    [[0.3, 0.3, 0.7, 0.7], [0, 0, 1, 1]],
    [[0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]],
    [[0, 0.5, 0.25, 0.25, 1, 1, 1, 0.75], [1, 0, 0, 0, 0, 1, 0, 1]]
  ];
  for (const [s, y] of cases) {
    assert.ok(Math.abs(rocAuc(s, y) - referenceAuc(s, y)) < 1e-12, JSON.stringify({ s, y }));
  }
  assert.equal(rocAuc([0.5, 0.5], [0, 1]), 0.5);
  assert.equal(rocAuc([0.5, 0.5], [1, 0]), 0.5);
  assert.equal(rocAuc([0.6, 0.5, 0.5], [1, 0, 1]), 0.75);
});

test('ROC-AUC matches reference over 500 randomized tie-heavy vectors', () => {
  let seed = 20270101;
  const rand = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let t = 0; t < 500; t++) {
    const n = 2 + Math.floor(rand() * 24);
    const s = Array.from({ length: n }, () => [0, 0.25, 0.5, 0.75, 1][Math.floor(rand() * 5)]);
    const y = Array.from({ length: n }, () => (rand() < 0.5 ? 1 : 0));
    if (!y.includes(1) || !y.includes(0)) continue;
    assert.ok(Math.abs(rocAuc(s, y) - referenceAuc(s, y)) < 1e-12);
  }
});

/* ------------------------------------------------------------------ */
/* Brier decomposition                                                  */
/* ------------------------------------------------------------------ */

test('Brier decomposition satisfies BS = REL - RES + UNC exactly', () => {
  const cases = [
    [[0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1]],
    [[0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1]],
    [[0.5, 0.5, 0.5], [0, 1, 1]],
    [[0, 1, 0.5], [0, 1, 1]]
  ];
  for (const [s, y] of cases) {
    const d = brierDecomposition(s, y);
    assert.ok(Math.abs(d.brier - (d.reliability - d.resolution + d.uncertainty)) < 1e-12, JSON.stringify(d));
    assert.ok(Math.abs(d.brier - brierScore(s, y)) < 1e-12);
    assert.ok(d.reliability >= -1e-12 && d.resolution >= -1e-12 && d.uncertainty >= 0);
  }
  const empty = brierDecomposition([], []);
  assert.deepEqual(empty, { brier: 0, reliability: 0, resolution: 0, uncertainty: 0 });
  const single = brierDecomposition([0.7], [1]);
  assert.ok(Number.isFinite(single.brier) && single.uncertainty === 0);
});

test('audit calibration exposes a valid held-out Brier decomposition', () => {
  for (const scenario of SCENARIOS) {
    const a = runAudit({ scenarioId: scenario.id });
    const d = a.calibration.testDecomposition;
    assert.ok(Math.abs(d.brier - (d.reliability - d.resolution + d.uncertainty)) < 1e-9, scenario.id);
    assert.ok(Math.abs(d.brier - a.baseline.metrics.brier) < 1e-9);
  }
});

/* ------------------------------------------------------------------ */
/* Subgroup PPV / NPV                                                   */
/* ------------------------------------------------------------------ */

test('subgroup PPV and NPV satisfy confusion-matrix identities', () => {
  const rows = [
    { g: 0, y: 1 }, { g: 0, y: 1 }, { g: 0, y: 0 }, { g: 0, y: 0 },
    { g: 1, y: 1 }, { g: 1, y: 0 }, { g: 1, y: 0 }, { g: 1, y: 0 }
  ];
  const scores = [0.9, 0.2, 0.8, 0.1, 0.9, 0.8, 0.2, 0.1];
  const groups = subgroupMetrics(rows, scores, 0.5, 'g');
  for (const g of [0, 1]) {
    const c = groups[g].counts;
    assert.ok(Math.abs(groups[g].ppv - (c.tp + c.fp ? c.tp / (c.tp + c.fp) : 0)) < 1e-12);
    assert.ok(Math.abs(groups[g].npv - (c.tn + c.fn ? c.tn / (c.tn + c.fn) : 0)) < 1e-12);
  }
  const f = fairnessSummary(groups, 'g');
  assert.ok(Math.abs(f.ppvDisparity - Math.abs(groups[0].ppv - groups[1].ppv)) < 1e-12);
  assert.ok(Math.abs(f.npvDisparity - Math.abs(groups[0].npv - groups[1].npv)) < 1e-12);
});

test('PPV and NPV are zero when their denominators are empty', () => {
  const groups = subgroupMetrics([{ g: 0, y: 1 }, { g: 1, y: 0 }], [0.1, 0.9], 0.5, 'g');
  assert.equal(groups[0].ppv, 0);
  assert.equal(groups[1].npv, 0);
});

test('exportAudit includes PPV and NPV gaps', () => {
  const rows = exportAudit(runAudit());
  for (const row of rows) {
    assert.ok(Number.isFinite(row.ppv_gap) && Number.isFinite(row.npv_gap));
  }
});

/* ------------------------------------------------------------------ */
/* CSV parsing and probability validation                               */
/* ------------------------------------------------------------------ */

test('parseCsv handles BOM, CRLF, embedded commas, escaped quotes and newlines', () => {
  const bom = String.fromCharCode(0xFEFF);
  const parsed = parseCsv(bom + 'score,label,note\r\n0.5,1,"a,b"\r\n0.2,0,"say ""hi"""\r\n0.1,0,"line1\nline2"\r\n');
  assert.deepEqual(parsed.headers, ['score', 'label', 'note']);
  assert.equal(parsed.rows.length, 3);
  assert.equal(parsed.rows[0].note, 'a,b');
  assert.equal(parsed.rows[1].note, 'say "hi"');
  assert.equal(parsed.rows[2].note, 'line1\nline2');
  assert.equal(parsed.skipped, 0);
});

test('parseCsv skips header-mismatched rows and empty trailing rows', () => {
  const parsed = parseCsv('a,b\n1,2\n3,4,5\n6,7\n\n');
  assert.equal(parsed.rows.length, 2);
  assert.equal(parsed.skipped, 1);
});

test('validateCsvRows rejects negative and above-one probabilities', () => {
  const rows = parseCsv('score,y\n0.5,1\n-0.1,0\n1.7,1\nabc,0\n1,0').rows;
  const result = validateCsvRows(rows, ['score', 'y'], ['score']);
  assert.equal(result.valid, false);
  assert.deepEqual(result.errors.map((e) => e.row), [2, 3, 4]);
  for (const e of result.errors) assert.match(e.message, /between 0 and 1/);
  const good = validateCsvRows(parseCsv('score,y\n0,0\n1,1\n0.5,1').rows, ['score'], ['score']);
  assert.equal(good.valid, true);
});

/* ------------------------------------------------------------------ */
/* Threshold fitting on empty/single-group inputs                       */
/* ------------------------------------------------------------------ */

test('fitGroupThresholds returns no phantom group for empty input', () => {
  assert.deepEqual(fitGroupThresholds([], [], { subgroupKey: 'g' }).thresholds, {});
  const one = fitGroupThresholds([{ g: 0, y: 1 }], [0.7], { subgroupKey: 'g' });
  assert.deepEqual(Object.keys(one.thresholds), ['0']);
});

test('adversarial small-split configuration stays finite and deterministic', () => {
  for (const opts of [
    { scenarioId: 'clinic', seed: 7, bias: 0.9, threshold: 0.8, n: 60 },
    { scenarioId: 'shelter', seed: 11, bias: 0.9, threshold: 0.1, n: 60 },
    { scenarioId: 'scholarship', seed: 3, bias: 0.9, threshold: 0.6, n: 100 }
  ]) {
    const a = runAudit(opts);
    const b = runAudit(opts);
    assert.deepEqual(a.strategies.map((s) => s.metrics.accuracy), b.strategies.map((s) => s.metrics.accuracy));
    for (const s of a.strategies) {
      for (const k of ['accuracy', 'recall', 'auc', 'brier', 'ece']) {
        assert.ok(Number.isFinite(s.metrics[k]), opts.scenarioId + ' ' + k);
      }
    }
  }
});

/* ------------------------------------------------------------------ */
/* UI invariants: demo wiring, tutorial story, no forbidden patterns    */
/* ------------------------------------------------------------------ */

test('demo button has a real scroll target and jumps without smooth scroll', () => {
  const html = read('index.html');
  const app = read('src/app.js');
  assert.match(html, /<section[^>]*id="hidden-failure"/);
  assert.match(app, /getElementById\('hidden-failure'\)/);
  assert.match(app, /window\.scrollTo\(/);
  assert.match(app, /behavior: 'auto'/);
  assert.doesNotMatch(app, /scrollIntoView\(\{[^}]*behavior: 'smooth'/);
});

test('demo renders a dramatic exact-number headline with no tagline under a title', () => {
  const html = read('index.html');
  const app = read('src/app.js');
  assert.match(html, /id="hidden-headline"/);
  assert.match(app, /hidden-headline'\)\.textContent/);
  assert.match(app, /pct\(harmed\.fnr\)/);
  assert.match(app, /pct\(gapBefore\)/);
  assert.match(app, /pct\(gapAfter\)/);
});

test('no em/en dashes, small all-caps eyebrows, or smooth-scroll rule', () => {
  for (const f of ['index.html', 'styles.css', 'src/app.js', 'src/charts.js', 'README.md']) {
    const text = read(f);
    assert.ok(!text.includes('—') && !text.includes('–'), f + ' contains an em/en dash');
  }
  const css = read('styles.css');
  assert.doesNotMatch(css, /scroll-behavior:\s*smooth/);
  const html = read('index.html');
  assert.doesNotMatch(html, /class="[^"]*(eyebrow|panel-kicker|hero-lead|section-intro|brand-tag|card-intro)/);
});

test('charts bind tooltips and the disparity legend wraps on narrow canvases', () => {
  const charts = read('src/charts.js');
  assert.match(charts, /function bindTooltip/);
  assert.match(charts, /bindTooltip\(canvas, entries\)/);
  assert.ok((charts.match(/bindTooltip\(canvas, entries\)/g) || []).length >= 3);
  assert.match(charts, /if \(lx \+ 18 \+ labelWidth \+ 10 > maxLx\)/);
});

test('README does not leak AI-tool mentions and documents the exact test command', () => {
  const readme = read('README.md');
  assert.doesNotMatch(readme, /\b(claude|gpt|llm|chatgpt|copilot|generative ai|artificial intelligence)\b/i);
  assert.match(readme, /node --test tests\/ml-engine\.test\.mjs/);
  assert.match(readme, /node --test tests\/wave2-regression\.test\.mjs/);
});
