/* Canvas charts for TriageLens. Pure drawing functions, no dependencies.
 * Every chart reads theme colors from CSS custom properties so it works in
 * both the light and dark palettes. */

function bindTooltip(canvas, entries) {
  if (!canvas || !entries || !entries.length) return;
  let tip = canvas._tip;
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'chart-tooltip';
    tip.hidden = true;
    const host = canvas.parentElement || document.body;
    host.appendChild(tip);
    canvas._tip = tip;
  }
  canvas.onmousemove = (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let best = null;
    let bestDistance = Infinity;
    for (const e of entries) {
      const d = (e.x - x) ** 2 + (e.y - y) ** 2;
      if (d < bestDistance) { bestDistance = d; best = e; }
    }
    if (!best || bestDistance > 26 ** 2) { tip.hidden = true; return; }
    tip.textContent = best.label;
    tip.hidden = false;
    tip.style.left = Math.min(rect.width - tip.offsetWidth - 4, Math.max(4, x + 12)) + 'px';
    tip.style.top = Math.max(4, y - tip.offsetHeight - 8) + 'px';
  };
  canvas.onmouseleave = () => { tip.hidden = true; };
}

function palette() {
  const cs = getComputedStyle(document.documentElement);
  const get = (name, fallback) => (cs.getPropertyValue(name) || fallback).trim();
  return {
    ink: get('--ink', '#16212c'),
    soft: get('--ink-soft', '#47586a'),
    faint: get('--ink-faint', '#607286'),
    panel: get('--panel', '#ffffff'),
    alt: get('--panel-alt', '#f0f3ed'),
    line: get('--line', '#d6ddd2'),
    accent: get('--accent', '#16624f'),
    accentSoft: get('--accent-soft', '#dfeee8'),
    clay: get('--clay', '#a2472c'),
    gold: get('--gold', '#9a7116'),
    blue: get('--blue', '#2c5a86')
  };
}

function setup(canvas) {
  if (!canvas) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(240, Math.round(rect.width || canvas.width));
  const h = Math.max(160, Math.round(rect.height || canvas.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, p: palette() };
}

function axes(ctx, x, y, w, h, p, xTicks, yTicks) {
  ctx.strokeStyle = p.line;
  ctx.lineWidth = 1;
  ctx.fillStyle = p.faint;
  ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of xTicks) {
    const px = x + t.v * w;
    ctx.beginPath();
    ctx.moveTo(px, y);
    ctx.lineTo(px, y + h);
    ctx.stroke();
    ctx.fillText(t.label, px, y + h + 6);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of yTicks) {
    const py = y + (1 - t.v) * h;
    ctx.beginPath();
    ctx.moveTo(x, py);
    ctx.lineTo(x + w, py);
    ctx.stroke();
    ctx.fillText(t.label, x - 7, py);
  }
  ctx.strokeStyle = p.soft;
  ctx.strokeRect(x, y, w, h);
}

const ticks = (n = 5) => Array.from({ length: n + 1 }, (_, i) => ({ v: i / n, label: (i / n).toFixed(1) }));

/** Receiver operating characteristic with AUC annotation. */
export function drawRoc(canvas, points, auc) {
  const s = setup(canvas);
  if (!s) return;
  const { ctx, w, h, p } = s;
  const m = { l: 42, r: 18, t: 18, b: 34 };
  const pw = w - m.l - m.r;
  const ph = h - m.t - m.b;
  axes(ctx, m.l, m.t, pw, ph, p, ticks(), ticks());

  ctx.fillStyle = p.faint;
  ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
  ctx.textAlign = 'center';
  ctx.fillText('False positive rate', m.l + pw / 2, h - 4);
  ctx.save();
  ctx.translate(11, m.t + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('True positive rate', 0, 0);
  ctx.restore();

  // Chance diagonal.
  ctx.strokeStyle = p.line;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(m.l, m.t + ph);
  ctx.lineTo(m.l + pw, m.t);
  ctx.stroke();
  ctx.setLineDash([]);

  if (!points || !points.length) return;
  ctx.beginPath();
  points.forEach((pt, i) => {
    const px = m.l + pt.fpr * pw;
    const py = m.t + (1 - pt.tpr) * ph;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = p.accent;
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  ctx.stroke();
  ctx.lineTo(m.l + pw, m.t + ph);
  ctx.closePath();
  ctx.fillStyle = p.accentSoft;
  ctx.globalAlpha = 0.55;
  ctx.fill();
  ctx.globalAlpha = 1;

  ctx.fillStyle = p.accent;
  ctx.font = '600 12px ' + getComputedStyle(document.body).fontFamily;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText('AUC ' + (auc ?? 0).toFixed(3), m.l + pw - 8, m.t + 8);

  const entries = points.map((pt) => ({
    x: m.l + pt.fpr * pw,
    y: m.t + (1 - pt.tpr) * ph,
    label: 'FPR ' + pt.fpr.toFixed(3) + ', TPR ' + pt.tpr.toFixed(3)
  }));
  bindTooltip(canvas, entries);
}

/** Ten bin reliability diagram with perfect-calibration reference. */
export function drawReliability(canvas, bins) {
  const s = setup(canvas);
  if (!s) return;
  const { ctx, w, h, p } = s;
  const m = { l: 42, r: 18, t: 18, b: 34 };
  const pw = w - m.l - m.r;
  const ph = h - m.t - m.b;
  axes(ctx, m.l, m.t, pw, ph, p, ticks(), ticks());

  ctx.fillStyle = p.faint;
  ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
  ctx.textAlign = 'center';
  ctx.fillText('Mean predicted probability', m.l + pw / 2, h - 4);
  ctx.save();
  ctx.translate(11, m.t + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Observed frequency', 0, 0);
  ctx.restore();

  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = p.line;
  ctx.beginPath();
  ctx.moveTo(m.l, m.t + ph);
  ctx.lineTo(m.l + pw, m.t);
  ctx.stroke();
  ctx.setLineDash([]);

  if (!bins || !bins.length) return;
  const bw = pw / bins.length;
  bins.forEach((b) => {
    if (!b.n) return;
    const px = m.l + b.confidence * pw;
    const py = m.t + (1 - b.observed) * ph;
    ctx.fillStyle = p.gold;
    ctx.globalAlpha = 0.72;
    ctx.beginPath();
    ctx.arc(px, py, Math.max(3, Math.min(8, 3 + Math.sqrt(b.n) * 0.35)), 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  });

  ctx.beginPath();
  bins.filter((b) => b.n).forEach((b, i) => {
    const px = m.l + b.confidence * pw;
    const py = m.t + (1 - b.observed) * ph;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = p.gold;
  ctx.lineWidth = 1.6;
  ctx.stroke();

  const entries = bins.filter((b) => b.n).map((b) => ({
    x: m.l + b.confidence * pw,
    y: m.t + (1 - b.observed) * ph,
    label: 'bin ' + b.lo.toFixed(1) + '-' + b.hi.toFixed(1) + ': predicted ' +
      b.confidence.toFixed(3) + ', observed ' + b.observed.toFixed(3) + ', n=' + b.n
  }));
  bindTooltip(canvas, entries);
}

/** Grouped bars: TPR and FNR per subgroup for each strategy. */
export function drawDisparity(canvas, strategies, groupLabels) {
  const s = setup(canvas);
  if (!s) return;
  const { ctx, w, h, p } = s;
  const m = { l: 42, r: 16, t: 30, b: 64 };
  const pw = w - m.l - m.r;
  const ph = h - m.t - m.b;
  axes(ctx, m.l, m.t, pw, ph, p, ticks(), ticks());

  const shown = strategies.filter((st) => st.id !== 'uncalibrated');
  if (!shown.length) return;
  const groupW = pw / shown.length;
  const barW = Math.min(28, (groupW - 20) / 4);
  const colors = [p.accent, p.clay, p.blue, p.gold];
  const entries = [];

  shown.forEach((strategy, si) => {
    const g = strategy.metrics.groups;
    const base = m.l + si * groupW + groupW / 2;
    const vals = [
      g.groupA.tpr, g.groupB.tpr,
      g.groupA.fnr, g.groupB.fnr
    ];
    vals.forEach((v, vi) => {
      const gx = base + (vi - 1.5) * (barW + 4);
      const gy = m.t + (1 - v) * ph;
      ctx.fillStyle = colors[vi];
      ctx.globalAlpha = vi < 2 ? 1 : 0.55;
      ctx.fillRect(gx - barW / 2, gy, barW, m.t + ph - gy);
      ctx.globalAlpha = 1;
      const names = ['TPR group 0', 'TPR group 1', 'FNR group 0', 'FNR group 1'];
      const groupNames = [(groupLabels && groupLabels[0]) || 'Group 0', (groupLabels && groupLabels[1]) || 'Group 1'];
      entries.push({
        x: gx,
        y: gy + (m.t + ph - gy) / 2,
        label: strategy.label + ' - ' + (vi < 2 ? 'TPR ' : 'FNR ') + groupNames[vi % 2] + ': ' +
          (v * 100).toFixed(1) + '% (' + names[vi] + ')'
      });
    });
    ctx.fillStyle = p.soft;
    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const short = strategy.id.replace('equal-', 'eq-').replace('proxy-removal', 'no-proxy').replace('baseline', 'global');
    ctx.fillText(short, base, m.t + ph + 8);
  });

  // Legend.
  const legend = [
    [p.accent, (groupLabels && groupLabels[0]) || 'Group 0 TPR'],
    [p.clay, (groupLabels && groupLabels[1]) || 'Group 1 TPR'],
    [p.blue, 'Group 0 FNR'],
    [p.gold, 'Group 1 FNR']
  ];
  let lx = m.l;
  let ly = 13;
  const maxLx = w - 8;
  legend.forEach(([color, label]) => {
    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    const labelWidth = ctx.measureText(label).width;
    if (lx + 18 + labelWidth + 10 > maxLx) { lx = m.l; ly += 14; }
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 5, 10, 10);
    ctx.fillStyle = p.faint;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, lx + 15, ly);
    lx += 18 + labelWidth + 18;
  });
  bindTooltip(canvas, entries);
}

/** Training loss curve. */
export function drawLoss(canvas, losses) {
  const s = setup(canvas);
  if (!s) return;
  const { ctx, w, h, p } = s;
  const m = { l: 48, r: 16, t: 16, b: 30 };
  const pw = w - m.l - m.r;
  const ph = h - m.t - m.b;
  axes(ctx, m.l, m.t, pw, ph, p, ticks(4), ticks(4));
  if (!losses || losses.length < 2) return;
  const lo = Math.min(...losses);
  const hi = Math.max(...losses);
  const span = hi - lo || 1;
  ctx.beginPath();
  losses.forEach((v, i) => {
    const px = m.l + (i / (losses.length - 1)) * pw;
    const py = m.t + (1 - (v - lo) / span) * ph;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = p.accent;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = p.faint;
  ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
  ctx.textAlign = 'center';
  ctx.fillText('epoch 0', m.l + 16, h - 3);
  ctx.fillText('epoch ' + (losses.length - 1) * 10, m.l + pw - 18, h - 3);
}
