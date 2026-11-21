/* Wave 4 regression tests: adversarial input hardening, CSV formula safety,
 * prototype-pollution defenses, honest metadata and export integrity.
 * Run: node --test tests/wave4-regression.test.mjs */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  generateDataset,
  runAudit,
  parseCsv,
  sanitizeRow,
  validateCsvRows,
  binStats,
  expectedCalibrationError,
  overallMetrics,
  fitGroupThresholds,
  csvCell,
  toCsv,
  exportAudit,
  trainDecisionTree,
  DATA_STATEMENT,
  MAX_N,
  DEFAULT_N
} from '../src/ml-engine.js';

const here = fileURLToPath(new URL('.', import.meta.url));

/* ---------------- fail-closed dataset options ---------------- */

test('generateDataset replaces a non-finite row count with the default', () => {
  for (const bad of [NaN, Infinity, -Infinity, 'lots']) {
    const d = generateDataset({ n: bad });
    assert.equal(d.n, DEFAULT_N, 'n=' + String(bad));
    assert.equal(d.train.length + d.test.length, DEFAULT_N);
  }
});

test('generateDataset clamps the row count into [20, MAX_N]', () => {
  assert.equal(generateDataset({ n: 3 }).n, 20);
  assert.equal(generateDataset({ n: -40 }).n, 20);
  const capped = generateDataset({ n: MAX_N + 12345, trainFraction: 0.9 });
  assert.equal(capped.n, MAX_N);
});

test('generateDataset clamps a hostile trainFraction into (0,1)', () => {
  for (const [given, lo, hi] of [[0, 0.05, 0.05], [1, 0.95, 0.95], [2, 0.95, 0.95], [-0.4, 0.05, 0.05], [NaN, 0.7, 0.7]]) {
    const d = generateDataset({ n: 200, trainFraction: given });
    assert.ok(d.trainFraction >= lo - 1e-9 && d.trainFraction <= hi + 1e-9, 'fraction ' + given + ' -> ' + d.trainFraction);
    assert.ok(d.train.length > 0 && d.test.length > 0);
  }
});

test('generateDataset records the effective clamped bias', () => {
  assert.equal(generateDataset({ bias: 7 }).bias, 1);
  assert.equal(generateDataset({ bias: -3 }).bias, 0);
});

test('runAudit never throws on hostile option shapes', () => {
  const a = runAudit({ n: NaN, bias: 'high', threshold: Infinity, trainFraction: NaN, seed: 'x', epochs: 5 });
  assert.equal(a.n, DEFAULT_N);
  assert.equal(a.bias, 0.6);
  assert.equal(a.threshold, 0.35);
  assert.ok(a.testCount > 0);
});

test('runAudit clamps an out-of-range threshold and reports the clamped value', () => {
  assert.equal(runAudit({ threshold: 1.7, epochs: 5 }).threshold, 1);
  assert.equal(runAudit({ threshold: -0.4, epochs: 5 }).threshold, 0);
  assert.equal(runAudit({ threshold: 'wide', epochs: 5 }).threshold, 0.35);
});

test('runAudit metadata reports the effective dataset, not the request', () => {
  const a = runAudit({ n: 8, trainFraction: 0.99, epochs: 5 });
  assert.equal(a.n, 20);
  assert.equal(a.trainFraction, 0.95);
  assert.equal(a.disclaimer, DATA_STATEMENT);
});

/* ---------------- decision tree on wide inputs ---------------- */

test('trainDecisionTree handles constant and missing feature columns', () => {
  const rows = Array.from({ length: 300 }, (_, i) => ({ f1: 7, f2: i % 3 ? 1 : undefined, y: i % 2 }));
  const t = trainDecisionTree(rows, { featureKeys: ['f1', 'f2'], minLeaf: 10 });
  assert.ok(t.leaves >= 1 && Number.isFinite(t.predictProba({ f1: 7, f2: 1 })));
});

test('trainDecisionTree ignores non-numeric feature values without NaN splits', () => {
  const rows = Array.from({ length: 200 }, (_, i) => ({ f: i, g: i % 2 ? 'x' : 5, y: i > 99 ? 1 : 0 }));
  const t = trainDecisionTree(rows, { featureKeys: ['f', 'g'], minLeaf: 8 });
  const p = t.predictProba({ f: 150, g: 'x' });
  assert.ok(p > 0.5, 'should route on the informative feature');
});

/* ---------------- metrics length mismatch ---------------- */

test('binStats ignores scores that have no paired label', () => {
  const b = binStats([0.1, 0.2, 0.9, 0.9], [1, 0], 4);
  assert.equal(b.reduce((s, g) => s + g.n, 0), 2);
});

test('expectedCalibrationError denominates on paired rows only', () => {
  const paired = expectedCalibrationError([0.9, 0.9], [1, 1]);
  const mismatched = expectedCalibrationError([0.9, 0.9, 0.0, 0.0], [1, 1]);
  assert.equal(paired, mismatched);
});

test('overallMetrics n and prevalence use paired length', () => {
  const m = overallMetrics([0.9, 0.1, 0.8], [1, 0]);
  assert.equal(m.n, 2);
  assert.equal(m.prevalence, 0.5);
});

/* ---------------- group thresholds ---------------- */

test('fitGroupThresholds returns an empty guards map for a single group', () => {
  const rows = Array.from({ length: 40 }, (_, i) => ({ rural: 0, y: i % 2 }));
  const fit = fitGroupThresholds(rows, rows.map((r) => r.y ? 0.8 : 0.2), { subgroupKey: 'rural' });
  assert.deepEqual(fit.guards, {});
  assert.equal(fit.thresholds[0], 0.5);
});

/* ---------------- CSV parsing hardening ---------------- */

test('parseCsv strips a UTF-8 byte order mark', () => {
  const { headers, rows } = parseCsv('﻿a,b\n1,2\n');
  assert.deepEqual(headers, ['a', 'b']);
  assert.equal(rows[0].a, 1);
});

test('parseCsv keeps a __proto__ column as data, never as a prototype', () => {
  const { rows } = parseCsv('__proto__,x\npolluted,1\n');
  assert.equal(Object.getPrototypeOf(rows[0]), Object.prototype);
  assert.equal(rows[0].__proto__, 'polluted');
  assert.equal({}.polluted, undefined);
});

test('sanitizeRow cannot pollute an object prototype', () => {
  const hostile = JSON.parse('{"__proto__":{"isAdmin":true},"x":5}');
  const out = sanitizeRow(hostile, { x: 0 });
  assert.equal(out.x, 5);
  assert.equal(out.isAdmin, undefined);
  assert.equal({}.isAdmin, undefined);
  assert.equal(Object.getPrototypeOf(out), Object.prototype);
  assert.equal(out.__proto__.isAdmin, true, 'proto data kept as a plain value');
});

test('parseCsv then validateCsvRows fails closed on wrong-width rows', () => {
  const { rows, skipped } = parseCsv('a,b\n1,2\n3\n4,5,6\n');
  assert.equal(skipped, 2);
  assert.equal(rows.length, 1);
  const v = validateCsvRows(rows, ['a', 'b'], []);
  assert.ok(v.valid);
});

/* ---------------- formula-safe CSV export ---------------- */

test('csvCell prefixes spreadsheet formula leaders', () => {
  for (const evil of ['=1+1', '+SUM(A1:A2)', '-10^3', '@cmd', '\t=1', '\r=1']) {
    const cell = csvCell(evil);
    const inner = cell.startsWith('"') ? cell.slice(1, -1).replace(/""/g, '"') : cell;
    assert.ok(inner.startsWith("'"), 'unescaped formula: ' + JSON.stringify(evil));
  }
});

test('csvCell quotes and doubles per RFC 4180', () => {
  assert.equal(csvCell('a,b'), '"a,b"');
  assert.equal(csvCell('say "hi"'), '"say ""hi"""');
  assert.equal(csvCell('line\nbreak'), '"line\nbreak"');
  assert.equal(csvCell(4.25), '4.25');
  assert.equal(csvCell(null), '');
  assert.equal(csvCell(NaN), '');
});

test('toCsv writes CRLF records that parseCsv reads back', () => {
  const rows = [{ a: 'x,y', b: 2 }, { a: 'plain', b: -3 }];
  const text = toCsv(rows);
  assert.ok(text.includes('\r\n'));
  const parsed = parseCsv(text);
  assert.equal(parsed.rows.length, 2);
  assert.equal(parsed.rows[0].a, 'x,y');
  assert.equal(parsed.rows[1].b, -3);
});

test('exported strategy CSV round-trips and contains no live formulas', () => {
  const audit = runAudit({ scenarioId: 'clinic', epochs: 20 });
  const text = toCsv(exportAudit(audit));
  const parsed = parseCsv(text);
  assert.equal(parsed.rows.length, audit.strategies.length);
  for (const field of parsed.rows.flatMap((r) => Object.values(r))) {
    const s = String(field);
    assert.ok(!/^[=+\-@\t\r]/.test(s) || s.startsWith("'"), 'unsafe cell ' + s);
  }
});

test('a forged audit row cannot smuggle a formula into the CSV', () => {
  const fake = {
    strategies: [{
      id: '=HYPERLINK("http://evil")',
      label: '@SUM(1)',
      threshold: 0.5,
      metrics: { accuracy: 0.5, precision: 0, recall: 0, f1: 0, auc: 0, brier: 0, ece: 0,
        groups: { demographicParityGap: 0, equalOpportunityGap: 0, fnrDisparity: 0, fprDisparity: 0, ppvDisparity: 0, npvDisparity: 0 } }
    }]
  };
  const text = toCsv(exportAudit(fake));
  const parsed = parseCsv(text);
  assert.ok(String(parsed.rows[0].strategy).startsWith("'"), parsed.rows[0].strategy);
  assert.ok(String(parsed.rows[0].label).startsWith("'"), parsed.rows[0].label);
});

/* ---------------- index.html security posture ---------------- */

test('index.html ships a restrictive content security policy', () => {
  const html = readFileSync(here + '../index.html', 'utf8');
  const csp = html.match(/Content-Security-Policy" content="([^"]+)"/);
  assert.ok(csp, 'CSP meta tag missing');
  for (const directive of ["default-src 'none'", "script-src 'self'", "connect-src 'none'", "object-src 'none'"]) {
    assert.ok(csp[1].includes(directive), 'missing ' + directive);
  }
});

test('index.html keeps the skip link target focusable', () => {
  const html = readFileSync(here + '../index.html', 'utf8');
  assert.match(html, /<main id="main" tabindex="-1">/);
});

test('no shipped file uses innerHTML, eval or document.write', () => {
  for (const f of ['../index.html', '../src/app.js', '../src/ml-engine.js', '../src/charts.js']) {
    const text = readFileSync(here + f, 'utf8');
    assert.ok(!/innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\s*\(|new Function/.test(text), f);
  }
});
