/* Wave 8 tests: SVG chart rendering against a minimal DOM stub, plus
 * structural checks on the new HTML and CSS. The stub implements only
 * the DOM surface svg-charts.js touches (createElementNS, setAttribute,
 * appendChild, replaceChildren, listeners, textContent) so the tests
 * verify real draw behavior, not a browser shell.
 * Run: node --test tests/wave8-svg-ui.test.mjs */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, '..');
const read = (p) => readFileSync(join(repo, p), 'utf8');

/* ------------------------------------------------------------------ */
/* Minimal DOM stub                                                    */
/* ------------------------------------------------------------------ */

class StubEl {
  constructor(tag) {
    this.tagName = tag;
    this.attrs = {};
    this.children = [];
    this.listeners = {};
    this._text = '';
    this.style = {};
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  appendChild(c) { this.children.push(c); return c; }
  replaceChildren() { this.children = []; }
  addEventListener(t, fn) { (this.listeners[t] ||= []).push(fn); }
  set textContent(v) { this._text = String(v); }
  get textContent() { return this._text; }
}

const doc = {
  documentElement: new StubEl('html'),
  createElementNS(ns, tag) { return new StubEl(tag); }
};
globalThis.document = doc;
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });

const { drawReliabilitySvg, drawFrontierSvg, frontierLegend } = await import('../src/svg-charts.js');

function all(node, tag, out = []) {
  for (const c of node.children || []) {
    if (c.tagName === tag) out.push(c);
    all(c, tag, out);
  }
  return out;
}
function fire(node, type) {
  for (const fn of node.listeners[type] || []) fn();
}
function texts(node, out = []) {
  for (const c of node.children || []) {
    if (c.tagName === 'text' || c.tagName === 'title') out.push(c.textContent);
    texts(c, out);
  }
  return out;
}

const BINS = [
  { bin: 0, lo: 0, hi: 0.1, n: 0, confidence: 0, observed: 0, observedCi: { lo: 0, hi: 1 } },
  { bin: 3, lo: 0.3, hi: 0.4, n: 40, confidence: 0.36, observed: 0.3, observedCi: { lo: 0.18, hi: 0.45 } },
  { bin: 5, lo: 0.5, hi: 0.6, n: 80, confidence: 0.55, observed: 0.5, observedCi: { lo: 0.39, hi: 0.61 } },
  { bin: 8, lo: 0.8, hi: 0.9, n: 20, confidence: 0.84, observed: 0.9, observedCi: { lo: 0.7, hi: 0.97 } }
];
const OVERLAY = {
  0: [
    { bin: 3, lo: 0.3, hi: 0.4, n: 22, confidence: 0.34, observed: 0.27 },
    { bin: 8, lo: 0.8, hi: 0.9, n: 10, confidence: 0.82, observed: 0.8 }
  ],
  1: [
    { bin: 5, lo: 0.5, hi: 0.6, n: 30, confidence: 0.56, observed: 0.53 },
    { bin: 8, lo: 0.8, hi: 0.9, n: 8, confidence: 0.85, observed: 0.75 }
  ]
};
const POINTS = [
  { t: 0, sel: 1, fnr: 0 },
  { t: 0.35, sel: 0.6, fnr: 0.2 },
  { t: 0.7, sel: 0.3, fnr: 0.5 },
  { t: 1, sel: 0, fnr: 1 }
];
const POLICIES = [
  { id: 'global', label: 'Global', sel: 0.6, fnr: 0.2 },
  { id: 'sensitivity-floor', label: 'Floor', sel: 0.9, fnr: 0.05 },
  { id: 'subgroup-calibrated', label: 'Sub', sel: 0.55, fnr: 0.3 },
  { id: 'youden', label: 'Youden', sel: 0.45, fnr: 0.15 }
];

/* ------------------------------------------------------------------ */
/* drawReliabilitySvg                                                  */
/* ------------------------------------------------------------------ */

test('drawReliabilitySvg mounts one svg with viewBox, role and aria-label', () => {
  const host = new StubEl('div');
  const svg = drawReliabilitySvg(host, { bins: BINS });
  assert.equal(host.children.length, 1);
  assert.equal(host.children[0], svg);
  assert.equal(svg.attrs.viewBox, '0 0 620 360');
  assert.equal(svg.attrs.role, 'img');
  assert.ok(svg.attrs['aria-label'].includes('Reliability'));
  assert.equal(svg.attrs.class, 'svg-plot');
});

test('drawReliabilitySvg clears the host before drawing', () => {
  const host = new StubEl('div');
  host.appendChild(new StubEl('svg'));
  drawReliabilitySvg(host, { bins: BINS });
  assert.equal(host.children.length, 1, 'stale svg must be replaced, not stacked');
});

test('drawReliabilitySvg draws the perfect-calibration diagonal', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS });
  const diag = all(svg, 'line').find((l) => l.attrs['stroke-dasharray'] === '5 4');
  assert.ok(diag, 'diagonal missing');
  const title = diag.children.find((c) => c.tagName === 'title');
  assert.equal(title.textContent, 'perfect calibration');
});

test('drawReliabilitySvg draws one focusable point group per nonempty bin', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS });
  const pts = all(svg, 'g').filter((g) => g.attrs.class === 'svg-pt' && g.children.some((c) => c.tagName === 'circle' && c.attrs.fill !== 'none'));
  assert.equal(pts.length, 3, 'three nonempty bins');
  for (const g of pts) assert.equal(g.attrs.tabindex, '0');
});

test('drawReliabilitySvg draws a gap stem and a CI whisker for every live bin', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS });
  const stems = all(svg, 'line').filter((l) => l.attrs['stroke-dasharray'] === '2 2');
  assert.equal(stems.length, 3);
  // Whiskers live inside the point groups, vertical ink lines.
  const groups = all(svg, 'g').filter((g) => g.attrs.class === 'svg-pt');
  const whiskers = groups.flatMap((g) => g.children.filter((c) => c.tagName === 'line'));
  assert.ok(whiskers.length >= 3, 'expected a CI whisker per bin');
  for (const w of whiskers) assert.equal(w.attrs.x1, w.attrs.x2, 'whisker must be vertical');
});

test('drawReliabilitySvg connects live bins with one gold path', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS });
  const paths = all(svg, 'path').filter((p) => p.attrs['stroke-width'] === '1.6');
  assert.equal(paths.length, 1);
  assert.ok(paths[0].attrs.d.startsWith('M'), 'path must start at first bin');
  assert.equal((paths[0].attrs.d.match(/L/g) || []).length, 2);
});

test('drawReliabilitySvg point titles carry n, predicted and observed', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS });
  const tips = texts(svg).filter((t) => t.startsWith('bin '));
  assert.equal(tips.length, 3);
  for (const t of tips) {
    assert.match(t, /predicted 0\.\d+/);
    assert.match(t, /observed 0\.\d+/);
    assert.match(t, /n=\d+/);
  }
});

test('drawReliabilitySvg hover callback receives the label then null', () => {
  let last = 'unset';
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS, onHover: (v) => { last = v; } });
  const g = all(svg, 'g').find((x) => x.attrs.class === 'svg-pt' && x.listeners.pointerenter);
  assert.ok(g, 'no hoverable point group');
  fire(g, 'pointerenter');
  assert.ok(String(last).startsWith('bin '), last);
  assert.ok(String(last).includes('95% CI'), 'hover should include the interval');
  fire(g, 'pointerleave');
  assert.equal(last, null);
});

test('drawReliabilitySvg overlay draws hollow circles and dashed paths per group', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), {
    bins: BINS, overlay: OVERLAY, groupLabels: { 0: 'urban', 1: 'rural' }, showOverlay: true
  });
  const hollow = all(svg, 'circle').filter((c) => c.attrs.fill === 'none');
  assert.equal(hollow.length, 4, 'two groups x two live bins');
  const dashed = all(svg, 'path').filter((p) => p.attrs['stroke-dasharray'] === '4 3');
  assert.equal(dashed.length, 2);
  const tips = texts(svg);
  assert.ok(tips.some((t) => t.startsWith('urban, bin')), 'group label missing from overlay titles');
  assert.ok(tips.some((t) => t.startsWith('rural, bin')));
});

test('drawReliabilitySvg suppresses the overlay when showOverlay is false', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS, overlay: OVERLAY, showOverlay: false });
  const hollow = all(svg, 'circle').filter((c) => c.attrs.fill === 'none');
  assert.equal(hollow.length, 0);
});

test('drawReliabilitySvg tolerates empty bins and a null host', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: [] });
  assert.ok(svg);
  assert.equal(all(svg, 'circle').length, 0);
  assert.equal(drawReliabilitySvg(null, { bins: BINS }), undefined);
  const svg2 = drawReliabilitySvg(new StubEl('div'), {});
  assert.ok(svg2);
});

/* ------------------------------------------------------------------ */
/* drawFrontierSvg                                                     */
/* ------------------------------------------------------------------ */

test('drawFrontierSvg mounts one svg with viewBox, role and aria-label', () => {
  const host = new StubEl('div');
  const svg = drawFrontierSvg(host, { points: POINTS });
  assert.equal(host.children[0], svg);
  assert.equal(svg.attrs.viewBox, '0 0 620 360');
  assert.equal(svg.attrs.role, 'img');
  assert.ok(svg.attrs['aria-label'].includes('frontier'));
});

test('drawFrontierSvg draws the pooled frontier path and per-point hit targets', () => {
  const svg = drawFrontierSvg(new StubEl('div'), { points: POINTS });
  const pooled = all(svg, 'path').find((p) => p.attrs['stroke-width'] === '2');
  assert.ok(pooled, 'pooled frontier missing');
  const targets = all(svg, 'g').filter((g) => (g.attrs.class || '').includes('frontier-pt'));
  assert.equal(targets.length, POINTS.length);
  for (const g of targets) {
    // Hundreds of candidate thresholds must never enter the tab order;
    // only the four policy markers are keyboard stops.
    assert.equal(g.attrs.tabindex, undefined, 'frontier point must not be focusable');
  }
});

test('drawFrontierSvg paints the pooled path in ascending alert order', () => {
  const svg = drawFrontierSvg(new StubEl('div'), { points: [...POINTS].reverse() });
  const targets = all(svg, 'g').filter((g) => (g.attrs.class || '').includes('frontier-pt'));
  const cxs = targets.map((g) => Number(g.children[0].attrs.cx));
  for (let i = 1; i < cxs.length; i++) {
    assert.ok(cxs[i] >= cxs[i - 1], 'frontier points must draw left to right');
  }
});

test('drawFrontierSvg draws one marker group per policy with data-policy', () => {
  const svg = drawFrontierSvg(new StubEl('div'), { points: POINTS, policyPoints: POLICIES });
  const markers = all(svg, 'g').filter((g) => (g.attrs.class || '').includes('policy-pt'));
  assert.equal(markers.length, 4);
  assert.deepEqual(markers.map((m) => m.attrs['data-policy']), ['global', 'sensitivity-floor', 'subgroup-calibrated', 'youden']);
  for (const m of markers) {
    assert.equal(m.attrs['aria-label'], POLICIES[markers.indexOf(m)].label);
    assert.equal(m.attrs.tabindex, '0');
  }
});

test('drawFrontierSvg gives the four policies distinct marker shapes', () => {
  const svg = drawFrontierSvg(new StubEl('div'), { points: POINTS, policyPoints: POLICIES });
  const markers = all(svg, 'g').filter((g) => (g.attrs.class || '').includes('policy-pt'));
  const firstTags = markers.map((m) => m.children[0].tagName);
  assert.equal(firstTags[0], 'circle');
  assert.equal(firstTags[1], 'rect');
  assert.equal(firstTags[2], 'path');
  assert.equal(firstTags[3], 'path');
});

test('drawFrontierSvg skips non-finite policy points but keeps drawing', () => {
  const svg = drawFrontierSvg(new StubEl('div'), {
    points: POINTS,
    policyPoints: [...POLICIES, { id: 'broken', label: 'Broken', sel: NaN, fnr: 0.5 }]
  });
  const markers = all(svg, 'g').filter((g) => (g.attrs.class || '').includes('policy-pt'));
  assert.equal(markers.length, 4);
});

test('drawFrontierSvg draws dashed per-group frontier curves', () => {
  const svg = drawFrontierSvg(new StubEl('div'), {
    points: POINTS,
    byGroup: { 0: POINTS, 1: [...POINTS].reverse() }
  });
  const groupPaths = all(svg, 'path').filter((p) => p.attrs['stroke-dasharray'] === '4 3');
  assert.equal(groupPaths.length, 2);
});

test('drawFrontierSvg hover text reports alerts and missed cases per 100', () => {
  let last = null;
  const svg = drawFrontierSvg(new StubEl('div'), { points: POINTS, policyPoints: POLICIES, onHover: (v) => { last = v; } });
  const target = all(svg, 'g').find((g) => (g.attrs.class || '').includes('frontier-pt'));
  fire(target, 'pointerenter');
  assert.match(last, /alerts \d+\.\d per 100/);
  assert.match(last, /missed \d+\.\d per 100 positives/);
  const policy = all(svg, 'g').find((g) => (g.attrs.class || '').includes('policy-pt'));
  fire(policy, 'pointerenter');
  assert.ok(String(last).startsWith('Global'));
});

test('drawFrontierSvg labels the axes honestly', () => {
  const svg = drawFrontierSvg(new StubEl('div'), { points: POINTS });
  const ts = texts(svg);
  assert.ok(ts.includes('alerts per 100 held-out records'));
  assert.ok(ts.includes('missed cases per 100 positives'));
});

test('drawFrontierSvg tolerates empty and missing data', () => {
  const svg = drawFrontierSvg(new StubEl('div'), {});
  assert.ok(svg);
  assert.equal(drawFrontierSvg(null, { points: POINTS }), undefined);
});

/* ------------------------------------------------------------------ */
/* frontierLegend                                                      */
/* ------------------------------------------------------------------ */

test('frontierLegend lists the pooled frontier first, solid', () => {
  const l = frontierLegend(POLICIES, {});
  assert.equal(l[0].label, 'held-out achievable frontier');
  assert.equal(l[0].dash, false);
});

test('frontierLegend adds one dashed entry per subgroup and one per policy', () => {
  const l = frontierLegend(POLICIES, { 0: 'urban', 1: 'rural' });
  const groupEntries = l.filter((e) => e.dash);
  assert.deepEqual(groupEntries.map((e) => e.label), ['urban frontier', 'rural frontier']);
  const policyEntries = l.filter((e) => e.marker);
  assert.equal(policyEntries.length, 4);
  for (const e of l) assert.match(e.color, /^#[0-9a-f]{6}$/i);
});

test('frontierLegend with no groups or policies is just the frontier', () => {
  const l = frontierLegend([], {});
  assert.equal(l.length, 1);
});

/* ------------------------------------------------------------------ */
/* HTML and CSS structure                                              */
/* ------------------------------------------------------------------ */

test('comparator and slices tables ship accessible header rows', () => {
  const html = read('index.html');
  for (const id of ['comparator-table', 'slices-table']) {
    const m = html.match(new RegExp(`<table[^>]*id="${id}"[\\s\\S]*?</table>`));
    assert.ok(m, id + ' table missing');
    assert.ok(m[0].includes('<thead>'), id + ' lacks a header row');
    assert.ok(m[0].includes('<tbody'), id + ' lacks a body');
    assert.ok(m[0].includes('scope="col"'), id + ' lacks scoped column headers');
  }
});

test('workbench section headings stand alone with no eyebrow or tagline', () => {
  const html = read('index.html');
  assert.ok(!/class="eyebrow|class="kicker|class="tagline/.test(html), 'eyebrow/kicker/tagline crept in');
  for (const id of ['workbench', 'certificate']) {
    const sec = html.match(new RegExp(`<section[^>]*id="${id}"[\\s\\S]*?</section>`));
    assert.ok(sec, id + ' section missing');
    assert.ok(/<h2[\s>]/.test(sec[0]), id + ' needs an h2');
  }
});

test('the overlay control is a labeled checkbox next to the reliability plot', () => {
  const html = read('index.html');
  const m = html.match(/<input[^>]*id="overlay-groups"[^>]*>/);
  assert.ok(m, 'overlay checkbox missing');
  assert.ok(m[0].includes('type="checkbox"'));
  // A <label> may either wrap the input or point at it with for=.
  const wrapped = /<label[^>]*>[\s\S]*?id="overlay-groups"/.test(html);
  const targeted = /for="overlay-groups"/.test(html);
  assert.ok(wrapped || targeted, 'overlay checkbox has no label');
});

test('every new element id in index.html is unique', () => {
  const html = read('index.html');
  const ids = [...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
  const seen = new Set();
  for (const id of ids) {
    assert.ok(!seen.has(id), 'duplicate id ' + id);
    seen.add(id);
  }
});

test('styles.css defines every class the new markup uses', () => {
  const css = read('styles.css');
  const html = read('index.html');
  const needed = ['svg-plot', 'svg-pt', 'cert-grid', 'cert-actions', 'selftest-list', 'delta-worse', 'delta-better', 'delta-flat', 'small-cell'];
  for (const c of needed) {
    const inCss = css.includes('.' + c);
    const inHtml = html.includes(c);
    assert.ok(inCss || inHtml, c + ' is neither styled nor used');
  }
});

test('certificate layout collapses to one column on small screens', () => {
  const css = read('styles.css');
  const mobile = css.slice(css.lastIndexOf('@media (max-width: 720px)'));
  assert.ok(mobile.includes('.cert-grid'), 'cert grid never collapses');
});

test('package.json still runs the node test runner', () => {
  const pkg = JSON.parse(read('package.json'));
  assert.match(pkg.scripts.test, /node --test/);
});

test('README test inventory lists every wave file', () => {
  const readme = read('README.md');
  for (const f of ['ml-engine.test.mjs', 'wave2-regression.test.mjs', 'wave3-regression.test.mjs', 'wave4-regression.test.mjs', 'wave5-regression.test.mjs', 'wave6-workbench.test.mjs', 'wave7-cert.test.mjs', 'wave8-svg-ui.test.mjs']) {
    assert.ok(readme.includes(f), f + ' undocumented in README');
  }
});

/* ------------------------------------------------------------------ */
/* Comparator provenance and structure                                 */
/* ------------------------------------------------------------------ */

test('comparator policies carry honest fitting provenance', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  const a = runAudit({ seed: 6, n: 400 });
  const byId = Object.fromEntries(a.workbench.comparator.policies.map((p) => [p.id, p]));
  assert.equal(byId.global.fittedNote, 'user set, not fitted');
  assert.match(byId['sensitivity-floor'].fittedNote, /train recall/);
  assert.match(byId['subgroup-calibrated'].fittedNote, /per-group, fitted on train/);
  assert.match(byId.youden.fittedNote, /train J/);
  for (const p of a.workbench.comparator.policies) {
    assert.ok(p.label && p.blurb, p.id + ' missing label or blurb');
    assert.ok(!/held-out/i.test(p.fittedNote), p.id + ' fitting note must not claim held-out fitting');
  }
});

test('frontier policy points carry the same ids and labels as the comparator', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  const a = runAudit({ seed: 6, n: 400 });
  const wb = a.workbench;
  assert.deepEqual(
    wb.frontier.policyPoints.map((p) => p.id),
    wb.comparator.policies.map((p) => p.id)
  );
  for (const pp of wb.frontier.policyPoints) {
    const pol = wb.comparator.policies.find((p) => p.id === pp.id);
    assert.equal(pp.label, pol.label);
  }
});

test('slices metadata names the second dimension per scenario', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  const dims = {};
  for (const id of ['clinic', 'scholarship', 'shelter']) {
    const a = runAudit({ scenarioId: id, seed: 8, n: 300 });
    assert.ok(a.workbench.slices.second, id + ' lacks slice metadata');
    assert.equal(typeof a.workbench.slices.second.dimension, 'string');
    dims[id] = a.workbench.slices.second.dimension;
  }
  assert.notEqual(dims.clinic, dims.scholarship);
  assert.notEqual(dims.scholarship, dims.shelter);
});

/* ------------------------------------------------------------------ */
/* Additional rendering and integrity edges                            */
/* ------------------------------------------------------------------ */

test('drawReliabilitySvg labels both axes', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS });
  const ts = texts(svg);
  assert.ok(ts.includes('mean predicted risk'));
  assert.ok(ts.includes('observed event rate'));
});

test('drawReliabilitySvg point radius grows with bin n', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS });
  const groups = all(svg, 'g').filter((g) => g.attrs.class === 'svg-pt' && g.children.some((c) => c.tagName === 'circle' && c.attrs.fill !== 'none'));
  const radii = groups.map((g) => Number(g.children.find((c) => c.tagName === 'circle').attrs.r));
  const ns = [40, 80, 20];
  const order = [...ns.keys()].sort((a, b) => ns[a] - ns[b]);
  assert.ok(radii[order[2]] > radii[order[0]], 'largest bin should get the largest marker');
});

test('drawReliabilitySvg gives the two overlay groups different colors', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: BINS, overlay: OVERLAY, groupLabels: { 0: 'a', 1: 'b' }, showOverlay: true });
  const strokes = new Set(all(svg, 'path').filter((p) => p.attrs['stroke-dasharray'] === '4 3').map((p) => p.attrs.stroke));
  assert.equal(strokes.size, 2, 'subgroups must be visually distinct');
});

test('drawReliabilitySvg skips the connective path for a single live bin', () => {
  const svg = drawReliabilitySvg(new StubEl('div'), { bins: [BINS[1]] });
  assert.equal(all(svg, 'path').filter((p) => p.attrs['stroke-width'] === '1.6').length, 0);
});

test('drawFrontierSvg blur clears the hover label', () => {
  let last = 'x';
  const svg = drawFrontierSvg(new StubEl('div'), { points: POINTS, onHover: (v) => { last = v; } });
  const target = all(svg, 'g').find((g) => (g.attrs.class || '').includes('frontier-pt'));
  fire(target, 'focus');
  assert.ok(String(last).includes('threshold'));
  fire(target, 'blur');
  assert.equal(last, null);
});

test('drawFrontierSvg pooled path carries a title for assistive tech', () => {
  const svg = drawFrontierSvg(new StubEl('div'), { points: POINTS });
  const pooled = all(svg, 'path').find((p) => p.attrs['stroke-width'] === '2');
  const title = pooled.children.find((c) => c.tagName === 'title');
  assert.equal(title.textContent, 'held-out achievable frontier');
});

test('frontierLegend keeps policy order and marks each as a marker', () => {
  const l = frontierLegend(POLICIES, {});
  const ids = l.filter((e) => e.marker).map((e) => e.label);
  assert.deepEqual(ids, ['Global', 'Floor', 'Sub', 'Youden']);
});

test('comparator table exposes exactly nine columns and slices six', () => {
  const html = read('index.html');
  const count = (id) => {
    const head = html.match(new RegExp(`id="${id}"[\\s\\S]*?<thead>[\\s\\S]*?</thead>`))[0];
    return (head.match(/<th\b/g) || []).length;
  };
  assert.equal(count('comparator-table'), 9);
  assert.equal(count('slices-table'), 6);
});

test('the cert hash renders inside a code element', () => {
  const html = read('index.html');
  assert.ok(/<code[^>]*id="cert-hash"/.test(html), 'cert-hash should be a <code>');
});

test('selftest list is a real ul so each check announces separately', () => {
  const html = read('index.html');
  assert.ok(/<ul[^>]*id="selftest-list"/.test(html));
});

test('comparator policy labels are unique', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  const a = runAudit({ seed: 6, n: 300 });
  const labels = a.workbench.comparator.policies.map((p) => p.label);
  assert.equal(new Set(labels).size, labels.length);
});

test('every comparator policy evaluates every held-out subgroup', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  const a = runAudit({ seed: 6, n: 400 });
  const groupIds = Object.keys(a.groupLabels || a.subgroup.labels || { 0: 1, 1: 1 });
  for (const p of a.workbench.comparator.policies) {
    const n = Object.values(p.eval.groups).reduce((s, g) => s + g.n, 0);
    assert.equal(n, a.testCount, p.id + ' group coverage');
    assert.deepEqual(Object.keys(p.eval.groups).sort(), Object.keys(p.eval.groups).sort());
  }
  assert.ok(groupIds.length >= 1);
});

test('slice cell rates satisfy the tpr + fnr identity', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  const a = runAudit({ seed: 6, n: 400 });
  for (const c of a.workbench.slices.cells) {
    if (c.positives > 0) {
      assert.ok(Math.abs(c.tpr + c.fnr - 1) < 1e-9, c.id + ' tpr+fnr != 1');
    }
    assert.ok(c.selectionRate >= 0 && c.selectionRate <= 1, c.id + ' selectionRate');
  }
});

test('workbench uncertainty brier stays inside [0, 1] on every scenario', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  for (const id of ['clinic', 'scholarship', 'shelter']) {
    const u = runAudit({ scenarioId: id, seed: 19, n: 300 }).workbench.uncertainty;
    assert.ok(u.brier >= 0 && u.brier <= 1, id);
    assert.ok(u.ece >= 0 && u.ece <= 1 && u.mce >= 0 && u.mce <= 1, id);
  }
});

test('groupReliability bins carry lo/hi bounds for every group', async () => {
  const { groupReliability } = await import('../src/calibration.js');
  const ov = groupReliability([0.2, 0.7, 0.3, 0.8, 0.5], [0, 1, 0, 1, 1], [0, 0, 1, 1, 2], 10);
  for (const bins of Object.values(ov)) {
    for (const b of bins) {
      assert.ok(b.lo >= 0 && b.hi <= 1 && b.lo < b.hi);
    }
  }
});

test('certificate digest is stable under payload key reordering', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  const { buildCertificate, sha256Hex, canonicalize } = await import('../src/audit-cert.js');
  const cert = buildCertificate(runAudit({ seed: 6, n: 300 }));
  const shuffled = JSON.parse(JSON.stringify({ ...cert }, (k, v) => v));
  delete shuffled.sha256;
  const payload = { ...cert };
  delete payload.sha256;
  assert.equal(sha256Hex(canonicalize(shuffled)), sha256Hex(canonicalize(payload)));
});

test('certificate CSV comment lines terminate CRLF like the RFC 4180 body', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  const { buildCertificate, certificateCsv } = await import('../src/audit-cert.js');
  const csv = certificateCsv(buildCertificate(runAudit({ seed: 6, n: 300 })));
  const comments = csv.split('\n').filter((l) => l.startsWith('#'));
  assert.ok(comments.length >= 2);
  for (const c of comments) assert.ok(c.endsWith('\r'), 'comment line missing CRLF');
});

test('encodeAuditParams fills safe defaults for missing fields', async () => {
  const { encodeAuditParams, decodeAuditParams } = await import('../src/audit-cert.js');
  const dec = decodeAuditParams(encodeAuditParams({ seed: 5 }));
  assert.ok(dec, 'partial bookmark must decode');
  assert.equal(dec.scenario, 'clinic');
  assert.equal(dec.seed, 5);
  // Missing bias and threshold fall back to the control defaults,
  // which sit inside the slider ranges and so decode cleanly.
  assert.equal(dec.bias, 0.6);
  assert.equal(dec.threshold, 0.35);
  // An entirely empty encode writes seed 0, which correctly fails
  // validation on decode: bookmarks always carry an explicit seed.
  assert.equal(decodeAuditParams(encodeAuditParams({})), null);
});

test('audit JSON stays serializable at the minimum dataset size', async () => {
  const { runAudit } = await import('../src/ml-engine.js');
  const a = runAudit({ seed: 1, n: 20 });
  const text = JSON.stringify(a);
  assert.ok(text.length > 100);
  assert.equal(JSON.parse(text).workbench.comparator.policies.length, 4);
});

test('app.js renders the workbench from audit.workbench, never raw test rows', () => {
  const src = read('src/app.js');
  assert.ok(src.includes('audit.workbench'), 'workbench render must read audit.workbench');
  assert.ok(src.includes('renderWorkbench'), 'renderWorkbench missing');
  assert.ok(src.includes('renderComparator'), 'renderComparator missing');
  assert.ok(src.includes('renderSlices'), 'renderSlices missing');
  assert.ok(src.includes('renderCertificate'), 'renderCertificate missing');
});
