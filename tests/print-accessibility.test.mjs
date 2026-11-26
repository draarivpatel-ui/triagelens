import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const HTML = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const CSS = readFileSync(new URL('../styles.css', import.meta.url), 'utf8');
const APP = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');

test('print report is a direct body child with live audit fields', () => {
  assert.match(HTML, /<section class="print-report" id="print-report"/);
  assert.ok(HTML.indexOf('</main>') < HTML.indexOf('id="print-report"'));
  for (const id of ['print-scenario', 'print-params', 'print-rows', 'print-accuracy', 'print-auc', 'print-brier', 'print-gap', 'print-recall', 'print-digest']) {
    assert.ok(HTML.includes(`id="${id}"`), id);
  }
});

test('print stylesheet owns exactly one letter page without screen chrome', () => {
  const printStart = CSS.indexOf('@media print');
  assert.notEqual(printStart, -1);
  const printBlock = CSS.slice(printStart);
  assert.match(CSS, /\.print-report \{ display: none; \}/);
  assert.match(printBlock, /@page \{ size: letter; margin: 0\.45in; \}/);
  assert.match(printBlock, /body > \*:not\(#print-report\) \{ display: none !important; \}/);
  assert.match(printBlock, /#print-report \{[\s\S]*?display: block !important;/);
  assert.match(printBlock, /grid-template-columns: 1fr 1fr/);
  assert.match(printBlock, /break-inside: avoid/);
});

test('print summary renders live metrics and certificate provenance', () => {
  assert.match(APP, /function renderPrintSummary\(audit\)/);
  assert.match(APP, /renderPrintSummary\(audit\);/);
  assert.match(APP, /state\.cert\.sha256\.slice\(0, 16\)/);
});

test('reduced motion and disclosure target contracts are explicit', () => {
  assert.match(CSS, /@media \(prefers-reduced-motion: reduce\) \{[\s\S]*?transition-duration: 0\.01ms !important/);
  assert.match(CSS, /\.tree-details summary \{[^}]*min-height: 24px/);
});
