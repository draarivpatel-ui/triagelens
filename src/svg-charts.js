/* Interactive SVG charts for the TriageLens workbench.
 * Zero dependencies, DOM API only (no markup string sinks), theme colors
 * read from the same CSS custom properties the canvas charts use.
 *
 * Two charts:
 *  - drawReliabilitySvg: held-out reliability diagram, predicted risk vs
 *    observed event rate, with a perfect-calibration diagonal, visible
 *    gap stems per bin, Wilson intervals and a per-subgroup overlay.
 *  - drawFrontierSvg: the held-out clinical-utility frontier, alert
 *    volume vs missed cases, with the four comparator policies marked.
 */

const SVGNS = 'http://www.w3.org/2000/svg';

function el(name, attrs, parent) {
  const node = document.createElementNS(SVGNS, name);
  if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function text(parent, x, y, content, attrs) {
  const t = el('text', { x, y, ...attrs }, parent);
  t.textContent = content;
  return t;
}

function palette() {
  const cs = getComputedStyle(document.documentElement);
  const get = (name, fallback) => (cs.getPropertyValue(name) || fallback).trim();
  return {
    ink: get('--ink', '#16212c'),
    soft: get('--ink-soft', '#47586a'),
    faint: get('--ink-faint', '#607286'),
    panel: get('--panel', '#ffffff'),
    line: get('--line', '#d6ddd2'),
    accent: get('--accent', '#16624f'),
    clay: get('--clay', '#a2472c'),
    gold: get('--gold', '#9a7116'),
    blue: get('--blue', '#2c5a86')
  };
}

const FONT = '10px ' + "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";

function axes(svg, m, w, h, p, xTicks, yTicks) {
  for (const t of xTicks) {
    const px = m.l + t.v * w;
    el('line', { x1: px, y1: m.t, x2: px, y2: m.t + h, stroke: p.line, 'stroke-width': 1 }, svg);
    text(svg, px, m.t + h + 14, t.label, { 'text-anchor': 'middle', fill: p.faint, 'font-size': 10 });
  }
  for (const t of yTicks) {
    const py = m.t + (1 - t.v) * h;
    el('line', { x1: m.l, y1: py, x2: m.l + w, y2: py, stroke: p.line, 'stroke-width': 1 }, svg);
    text(svg, m.l - 7, py + 3, t.label, { 'text-anchor': 'end', fill: p.faint, 'font-size': 10 });
  }
  el('rect', { x: m.l, y: m.t, width: w, height: h, fill: 'none', stroke: p.soft, 'stroke-width': 1 }, svg);
}

const TICKS = [0, 0.25, 0.5, 0.75, 1].map((v) => ({ v, label: v.toFixed(2).replace(/0+$/, '0') }));
const TICKS100 = [0, 20, 40, 60, 80, 100].map((v) => ({ v: v / 100, label: String(v) }));

/**
 * Reliability diagram. `bins` are the workbench reliability rows
 * ({lo, hi, n, confidence, observed, observedCi}); `overlay` maps a
 * subgroup id to the same rows for a hollow per-group series.
 */
export function drawReliabilitySvg(host, { bins, overlay, groupLabels = {}, showOverlay = true, onHover } = {}) {
  if (!host) return;
  host.replaceChildren();
  const p = palette();
  const W = 620;
  const H = 360;
  const m = { l: 46, r: 18, t: 20, b: 38 };
  const pw = W - m.l - m.r;
  const ph = H - m.t - m.b;
  const svg = el('svg', {
    viewBox: '0 0 ' + W + ' ' + H,
    class: 'svg-plot',
    role: 'img',
    'aria-label': 'Reliability diagram: mean predicted risk against observed event rate per bin'
  }, host);
  svg.style.fontFamily = 'inherit';

  const X = (v) => m.l + v * pw;
  const Y = (v) => m.t + (1 - v) * ph;
  const hover = (node, label) => {
    if (!onHover) return;
    node.addEventListener('pointerenter', () => onHover(label));
    node.addEventListener('focus', () => onHover(label));
    node.addEventListener('pointerleave', () => onHover(null));
    node.addEventListener('blur', () => onHover(null));
  };

  axes(svg, m, pw, ph, p, TICKS, TICKS);
  text(svg, m.l + pw / 2, H - 6, 'mean predicted risk', { 'text-anchor': 'middle', fill: p.faint, 'font-size': 10 });
  const yLab = text(svg, 12, m.t + ph / 2, 'observed event rate', { 'text-anchor': 'middle', fill: p.faint, 'font-size': 10 });
  yLab.setAttribute('transform', 'rotate(-90 12 ' + (m.t + ph / 2) + ')');

  // Perfect calibration diagonal.
  const diag = el('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: p.faint, 'stroke-dasharray': '5 4', 'stroke-width': 1 }, svg);
  el('title', {}, diag).textContent = 'perfect calibration';

  const rows = Array.isArray(bins) ? bins : [];
  const live = rows.filter((b) => b.n > 0);

  // Visible calibration error: a stem from each point to the diagonal.
  for (const b of live) {
    el('line', {
      x1: X(b.confidence), y1: Y(b.observed),
      x2: X(b.confidence), y2: Y(b.confidence),
      stroke: b.observed >= b.confidence ? p.accent : p.clay,
      'stroke-width': 1.4, 'stroke-dasharray': '2 2', opacity: 0.65
    }, svg);
  }
  // Connect the bin centers so the curve reads as one shape.
  if (live.length > 1) {
    el('path', {
      d: live.map((b, i) => (i ? 'L' : 'M') + X(b.confidence) + ' ' + Y(b.observed)).join(' '),
      fill: 'none', stroke: p.gold, 'stroke-width': 1.6
    }, svg);
  }
  const maxN = Math.max(1, ...live.map((b) => b.n));
  for (const b of live) {
    const g = el('g', { tabindex: '0', class: 'svg-pt' }, svg);
    const r = 3 + 6 * Math.sqrt(b.n / maxN);
    el('circle', { cx: X(b.confidence), cy: Y(b.observed), r, fill: p.gold, 'fill-opacity': 0.8, stroke: p.ink, 'stroke-width': 0.5 }, g);
    if (b.observedCi) {
      el('line', {
        x1: X(b.confidence), y1: Y(b.observedCi.lo),
        x2: X(b.confidence), y2: Y(b.observedCi.hi),
        stroke: p.ink, 'stroke-width': 1.2, opacity: 0.55
      }, g);
    }
    const label = 'bin ' + b.lo.toFixed(1) + ' to ' + b.hi.toFixed(1) + ': predicted ' + b.confidence.toFixed(3) +
      ', observed ' + b.observed.toFixed(3) + ', n=' + b.n +
      (b.observedCi ? ', 95% CI ' + b.observedCi.lo.toFixed(2) + ' to ' + b.observedCi.hi.toFixed(2) : '');
    el('title', {}, g).textContent = label;
    g.setAttribute('role', 'img');
    g.setAttribute('aria-label', label);
    hover(g, label);
  }

  // Per-subgroup overlay: hollow markers at each group's own bin means.
  if (showOverlay && overlay && typeof overlay === 'object') {
    const colors = [p.accent, p.clay, p.blue, p.ink];
    Object.keys(overlay).forEach((g, gi) => {
      const color = colors[gi % colors.length];
      const name = groupLabels[g] || 'group ' + g;
      const gb = (overlay[g] || []).filter((b) => b.n > 0);
      if (gb.length > 1) {
        el('path', {
          d: gb.map((b, i) => (i ? 'L' : 'M') + X(b.confidence) + ' ' + Y(b.observed)).join(' '),
          fill: 'none', stroke: color, 'stroke-width': 1.2, 'stroke-dasharray': '4 3', opacity: 0.8
        }, svg);
      }
      for (const b of gb) {
        const gg = el('g', { tabindex: '0', class: 'svg-pt' }, svg);
        el('circle', { cx: X(b.confidence), cy: Y(b.observed), r: 4, fill: 'none', stroke: color, 'stroke-width': 1.8 }, gg);
        const label = name + ', bin ' + b.lo.toFixed(1) + ' to ' + b.hi.toFixed(1) + ': predicted ' +
          b.confidence.toFixed(3) + ', observed ' + b.observed.toFixed(3) + ', n=' + b.n;
        el('title', {}, gg).textContent = label;
        gg.setAttribute('role', 'img');
        gg.setAttribute('aria-label', label);
        hover(gg, label);
      }
    });
  }
  return svg;
}

/**
 * Clinical-utility frontier. x is alerts per 100 held-out records,
 * y is missed positives per 100 held-out positives. The overall curve
 * is the achievable frontier; dashed curves are per-subgroup frontiers;
 * markers are the four comparator policies.
 */
export function drawFrontierSvg(host, { points, byGroup, groupLabels = {}, policyPoints = [], onHover } = {}) {
  if (!host) return;
  host.replaceChildren();
  const p = palette();
  const W = 620;
  const H = 360;
  const m = { l: 46, r: 18, t: 20, b: 38 };
  const pw = W - m.l - m.r;
  const ph = H - m.t - m.b;
  const svg = el('svg', {
    viewBox: '0 0 ' + W + ' ' + H,
    class: 'svg-plot',
    role: 'img',
    'aria-label': 'Clinical utility frontier: alerts per 100 records against missed positives per 100'
  }, host);
  svg.style.fontFamily = 'inherit';

  const X = (v) => m.l + v * pw;
  const Y = (v) => m.t + (1 - v) * ph;
  const hover = (node, label) => {
    if (!onHover) return;
    node.addEventListener('pointerenter', () => onHover(label));
    node.addEventListener('focus', () => onHover(label));
    node.addEventListener('pointerleave', () => onHover(null));
    node.addEventListener('blur', () => onHover(null));
  };

  axes(svg, m, pw, ph, p, TICKS100, TICKS100);
  text(svg, m.l + pw / 2, H - 6, 'alerts per 100 held-out records', { 'text-anchor': 'middle', fill: p.faint, 'font-size': 10 });
  const yLab = text(svg, 12, m.t + ph / 2, 'missed cases per 100 positives', { 'text-anchor': 'middle', fill: p.faint, 'font-size': 10 });
  yLab.setAttribute('transform', 'rotate(-90 12 ' + (m.t + ph / 2) + ')');

  // Per-subgroup frontiers first so the pooled curve draws on top.
  const groupColors = [p.accent, p.clay, p.blue, p.ink];
  if (byGroup && typeof byGroup === 'object') {
    Object.keys(byGroup).forEach((g, gi) => {
      const color = groupColors[gi % groupColors.length];
      const pts = (byGroup[g] || []).slice().sort((a, b) => a.sel - b.sel);
      if (!pts.length) return;
      el('path', {
        d: pts.map((b, i) => (i ? 'L' : 'M') + X(b.sel) + ' ' + Y(b.fnr)).join(' '),
        fill: 'none', stroke: color, 'stroke-width': 1.2, 'stroke-dasharray': '4 3', opacity: 0.75
      }, svg);
    });
  }

  const pts = (Array.isArray(points) ? points : []).slice().sort((a, b) => a.sel - b.sel);
  if (pts.length) {
    const path = el('path', {
      d: pts.map((b, i) => (i ? 'L' : 'M') + X(b.sel) + ' ' + Y(b.fnr)).join(' '),
      fill: 'none', stroke: p.ink, 'stroke-width': 2
    }, svg);
    el('title', {}, path).textContent = 'held-out achievable frontier';
    for (const b of pts) {
      // No tabindex: hundreds of candidate thresholds would bury the four
      // policy markers in the tab order. Pointer hover still reads each.
      const g = el('g', { class: 'svg-pt frontier-pt' }, svg);
      el('circle', { cx: X(b.sel), cy: Y(b.fnr), r: 5, fill: 'transparent', 'pointer-events': 'all' }, g);
      const label = 'threshold ' + b.t.toFixed(3) + ': alerts ' + (b.sel * 100).toFixed(1) +
        ' per 100, missed ' + (b.fnr * 100).toFixed(1) + ' per 100 positives';
      el('title', {}, g).textContent = label;
      hover(g, label);
    }
  }

  // Policy markers with distinct shapes so color is never the only cue.
  const shapes = [
    (g, x, y, c) => el('circle', { cx: x, cy: y, r: 6, fill: c, stroke: p.panel, 'stroke-width': 1.5 }, g),
    (g, x, y, c) => el('rect', { x: x - 5.5, y: y - 5.5, width: 11, height: 11, fill: c, stroke: p.panel, 'stroke-width': 1.5 }, g),
    (g, x, y, c) => el('path', { d: 'M' + x + ' ' + (y - 7) + ' L' + (x + 6.5) + ' ' + (y + 5) + ' L' + (x - 6.5) + ' ' + (y + 5) + ' Z', fill: c, stroke: p.panel, 'stroke-width': 1.5 }, g),
    (g, x, y, c) => el('path', { d: 'M' + x + ' ' + (y - 7) + ' L' + (x + 6.5) + ' ' + y + ' L' + x + ' ' + (y + 7) + ' L' + (x - 6.5) + ' ' + y + ' Z', fill: c, stroke: p.panel, 'stroke-width': 1.5 }, g)
  ];
  const policyColors = { global: p.accent, 'sensitivity-floor': p.clay, 'subgroup-calibrated': p.gold, youden: p.blue };
  (Array.isArray(policyPoints) ? policyPoints : []).forEach((pp, i) => {
    if (!Number.isFinite(pp.sel) || !Number.isFinite(pp.fnr)) return;
    const g = el('g', { tabindex: '0', class: 'svg-pt policy-pt', 'data-policy': pp.id }, svg);
    shapes[i % shapes.length](g, X(pp.sel), Y(pp.fnr), policyColors[pp.id] || p.ink);
    const label = pp.label + ': alerts ' + (pp.sel * 100).toFixed(1) +
      ' per 100, missed ' + (pp.fnr * 100).toFixed(1) + ' per 100 positives';
    el('title', {}, g).textContent = label;
    g.setAttribute('role', 'img');
    g.setAttribute('aria-label', label);
    hover(g, label);
  });
  return svg;
}

/** Legend entries for the frontier chart, for the HTML legend list. */
export function frontierLegend(policyPoints, groupLabels = {}) {
  const p = palette();
  const out = [];
  const groupColors = [p.accent, p.clay, p.blue, p.ink];
  const policyColors = { global: p.accent, 'sensitivity-floor': p.clay, 'subgroup-calibrated': p.gold, youden: p.blue };
  out.push({ label: 'held-out achievable frontier', color: p.ink, dash: false });
  Object.keys(groupLabels).forEach((g, i) => {
    out.push({ label: groupLabels[g] + ' frontier', color: groupColors[i % groupColors.length], dash: true });
  });
  for (const pp of policyPoints || []) {
    out.push({ label: pp.label, color: policyColors[pp.id] || p.ink, marker: true });
  }
  return out;
}
