import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const CSS = readFileSync(new URL('../styles.css', import.meta.url), 'utf8');

test('interactive control borders use the high contrast control line token', () => {
  assert.match(CSS, /--control-line: #657764;/);
  assert.match(CSS, /html\[data-theme='dark'\][\s\S]*?--control-line: #6f8c7a;/);
  assert.match(CSS, /\.icon-button \{[\s\S]*?border: 1px solid var\(--control-line\)/);
  assert.match(CSS, /\.button\.ghost \{[^}]*border-color: var\(--control-line\)/);
  assert.match(CSS, /\.tree-details summary \{[^}]*border: 1px solid var\(--control-line\)/);
  assert.match(CSS, /\.case-card \{[^}]*border: 1px solid var\(--control-line\)/);
  assert.match(CSS, /html\[data-theme='dark'\] \.button\.primary \{ background: #0d4a3b; color: #eaf0ea; border-color: var\(--control-line\); \}/);
  const inputBlock = CSS.slice(CSS.indexOf('.select, .number-input {'), CSS.indexOf('.select:focus-visible'));
  assert.match(inputBlock, /border: 1px solid var\(--control-line\)/);
});

test('mobile interactive controls reserve a 44px touch height', () => {
  const start = CSS.indexOf('@media (max-width: 720px)');
  assert.notEqual(start, -1);
  const mobile = CSS.slice(start, CSS.indexOf('@media', start + 1) === -1 ? CSS.length : CSS.indexOf('@media', start + 1));
  assert.match(mobile, /\.icon-button \{ width: 44px; height: 44px; \}/);
  assert.match(mobile, /\.button, \.button\.small \{ min-height: 44px; \}/);
  assert.match(mobile, /\.select, \.number-input \{ min-height: 44px; \}/);
  assert.match(mobile, /\.range \{ min-height: 44px; \}/);
  assert.match(mobile, /\.check-line\.check-line \{ min-height: 44px; align-items: center; \}/);
  assert.match(mobile, /\.tree-details summary \{ min-height: 44px; \}/);
});
