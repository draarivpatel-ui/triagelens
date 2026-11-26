/* TriageLens UI controller. ES module, zero dependencies, no network access. */
import { runAudit, exportAudit, toCsv, SCENARIOS, formatGapChange, tradeoffNotes, DATA_STATEMENT } from './ml-engine.js';
import { drawRoc, drawReliability, drawDisparity, drawLoss } from './charts.js';
import { drawReliabilitySvg, drawFrontierSvg, frontierLegend } from './svg-charts.js';
import {
  buildCertificate,
  certificateJson,
  certificateCsv,
  certificateMarkdown,
  encodeAuditParams,
  decodeAuditParams,
  runSelfTests
} from './audit-cert.js';

const $ = (id) => document.getElementById(id);
const state = { audit: null, cert: null, timer: null };
const pct = (x) => (x * 100).toFixed(1) + '%';
const num = (x, d = 3) => Number(x).toFixed(d);

function scenarioNote(id) {
  const s = SCENARIOS.find((x) => x.id === id);
  return s ? s.outcomeNote : '';
}

function scheduleAudit() {
  if (state.timer) clearTimeout(state.timer);
  state.timer = setTimeout(run, 90);
}

function run() {
  const scenarioId = $('scenario').value;
  const bias = Number($('bias').value);
  const threshold = Number($('threshold').value);
  const seed = Math.min(999999999, Math.max(1, Math.floor(Number($('seed').value) || 1)));
  // Write the clamped value back so the field, the run, the URL hash and
  // any copied share link all carry the seed the audit actually used.
  $('seed').value = String(seed);

  $('bias-value').textContent = bias.toFixed(2);
  $('threshold-value').textContent = threshold.toFixed(2);
  $('seed-value').textContent = String(seed);
  $('scenario-note').textContent = scenarioNote(scenarioId);

  const t0 = performance.now();
  const audit = runAudit({ scenarioId, bias, threshold, seed });
  const wallMs = performance.now() - t0;
  state.audit = audit;
  state.cert = buildCertificate(audit);
  updateHash({ scenario: scenarioId, bias, threshold, seed });

  renderHero(audit, wallMs);
  renderMetrics(audit);
  renderStrategies(audit);
  renderWeights(audit);
  renderTree(audit);
  renderMiniStats(audit, wallMs);
  renderCounterexamples(audit);
  renderEthics(audit);
  renderCharts(audit);
  renderHidden(audit);
  renderWorkbench(audit);
  renderPrintSummary(audit);
  renderCertificate();
}

function renderHero(audit, wallMs) {
  $('hero-gap').textContent = pct(audit.baseline.metrics.groups.equalOpportunityGap);
  $('hero-auc').textContent = num(audit.models.lr.auc);
  $('hero-time').textContent = Math.round(wallMs) + ' ms';
  $('hero-scenario').textContent =
    'Synthetic scenario: ' + audit.scenarioName + ' | ' + audit.trainCount + ' train / ' + audit.testCount + ' held out.';
}

function renderMetrics(audit) {
  const b = audit.baseline.metrics;
  const eo = audit.strategies.find((s) => s.id === 'equal-opportunity');
  $('m-accuracy').textContent = pct(b.accuracy);
  $('m-auc').textContent = num(b.auc);
  $('m-brier').textContent = num(b.brier);
  $('m-brier-foot').textContent = 'raw ' + num(audit.calibration.testRawBrier) + ' → calibrated';
  $('m-gap').textContent = pct(b.groups.equalOpportunityGap) + ' → ' + pct(eo.metrics.groups.equalOpportunityGap);
  const drop = b.groups.equalOpportunityGap - eo.metrics.groups.equalOpportunityGap;
  $('m-gap-foot').textContent = formatGapChange(drop, eo.metrics.groups.equalOpportunityGap) + ' with equal opportunity';
  const dropPct = (from, to) => ((to - from) * 100).toFixed(1);
  const notes = tradeoffNotes(audit).map((n) => n.label + ': ' + n.changes.join(', '));
  $('recall-banner').textContent = notes.length
    ? 'Trade-off watch: ' + notes.join('; ') + '. Recall change vs global threshold: ' +
      'equal opportunity ' + dropPct(b.recall, eo.metrics.recall) + ' pp, selection ' +
      dropPct(b.recall, audit.strategies.find((s) => s.id === 'equal-selection').metrics.recall) + ' pp, proxy removal ' +
      dropPct(b.recall, audit.strategies.find((s) => s.id === 'proxy-removal').metrics.recall) + ' pp.'
    : 'Trade-off watch: no mitigation changes overall recall or widens a secondary gap at this setting.';
}

function renderStrategies(audit) {
  const tbody = $('strategy-rows');
  tbody.replaceChildren();
  const eos = audit.strategies.map((s) => s.metrics.groups.equalOpportunityGap);
  const bestEo = Math.min(...eos);
  const worstEo = Math.max(...eos);

  for (const s of audit.strategies) {
    const tr = document.createElement('tr');
    const nameCell = document.createElement('td');
    nameCell.className = 'policy-name';
    nameCell.append(document.createTextNode(s.label));
    const small = document.createElement('small');
    small.textContent = s.blurb;
    nameCell.append(small);
    tr.append(nameCell);

    const thresholdText = s.thresholds
      ? Object.entries(s.thresholds).map(([g, t]) => (audit.groupLabels[g] || 'G' + g) + ': ' + t.toFixed(2)).join(' | ')
      : Number(s.threshold).toFixed(2);
    const cells = [
      thresholdText,
      pct(s.metrics.accuracy),
      pct(s.metrics.recall),
      pct(s.metrics.groups.equalOpportunityGap),
      pct(s.metrics.groups.fnrDisparity),
      pct(s.metrics.groups.demographicParityGap)
    ];
    cells.forEach((value, i) => {
      const td = document.createElement('td');
      td.textContent = value;
      if (i === 3) {
        if (Math.abs(s.metrics.groups.equalOpportunityGap - bestEo) < 1e-9) td.className = 'best';
        else if (Math.abs(s.metrics.groups.equalOpportunityGap - worstEo) < 1e-9) td.className = 'worst';
      }
      tr.append(td);
    });
    tbody.append(tr);
  }

  const base = audit.baseline.metrics;
  const eo = audit.strategies.find((s) => s.id === 'equal-opportunity').metrics;
  const accuracyCost = base.accuracy - eo.accuracy;
  const gapDrop = base.groups.equalOpportunityGap - eo.groups.equalOpportunityGap;
  $('tradeoff-note').textContent =
    'Equal opportunity changes the EO gap from ' + pct(base.groups.equalOpportunityGap) + ' to ' +
    pct(eo.groups.equalOpportunityGap) + ' (' + formatGapChange(gapDrop, eo.groups.equalOpportunityGap) +
    ') for an accuracy change of ' + (accuracyCost >= 0 ? '-' : '+') + pct(Math.abs(accuracyCost)) +
    '. Recall changes and any widening secondary gaps are listed in the trade-off banner above.';
}

function renderWeights(audit) {
  const wrap = $('weight-bars');
  wrap.replaceChildren();
  const values = Object.entries(audit.models.lr.coefficients);
  const max = Math.max(0.001, ...values.map(([, v]) => Math.abs(v.standardized)));
  for (const [key, coef] of values) {
    const feature = audit.features.find((f) => f.key === key);
    const row = document.createElement('div');
    row.className = 'weight-row';
    const label = document.createElement('span');
    label.textContent = feature ? feature.label : key;
    label.title = label.textContent;
    const track = document.createElement('span');
    track.className = 'weight-track';
    const center = document.createElement('span');
    center.className = 'weight-center';
    const fill = document.createElement('span');
    fill.className = 'weight-fill ' + (coef.standardized >= 0 ? 'pos' : 'neg');
    const width = (Math.abs(coef.standardized) / max) * 50;
    fill.style.width = width + '%';
    if (coef.standardized < 0) fill.style.left = (50 - width) + '%';
    track.append(center, fill);
    const value = document.createElement('span');
    value.className = 'weight-value';
    value.textContent = (coef.standardized >= 0 ? '+' : '') + coef.standardized.toFixed(2);
    row.append(label, track, value);
    wrap.append(row);
  }
}

function renderTree(audit) {
  const wrap = $('tree-rules');
  wrap.replaceChildren();
  for (const rule of audit.models.tree.rules.slice(0, 12)) {
    const div = document.createElement('div');
    div.className = 'tree-rule';
    const cond = rule.conditions.length
      ? rule.conditions.map((c) => {
          const f = audit.features.find((x) => x.key === c.key);
          return (f ? f.key : c.key) + ' ' + c.op + ' ' + c.threshold.toFixed(1);
        }).join(' AND ')
      : 'all rows';
    div.textContent = cond + ' → p=' + rule.probability.toFixed(3) + ' (n=' + rule.n + ')';
    wrap.append(div);
  }
}

function renderMiniStats(audit, wallMs) {
  const wrap = $('mini-stats');
  wrap.replaceChildren();
  const items = [
    ['train rows', audit.trainCount],
    ['held-out rows', audit.testCount],
    ['tree leaves', audit.models.tree.leaves],
    ['tree depth', audit.models.tree.depth],
    ['temperature', num(audit.calibration.temperature, 2)],
    ['train ECE', num(audit.calibration.rawEce) + ' → ' + num(audit.calibration.calibratedEce)],
    ['train Brier', num(audit.calibration.rawBrier) + ' → ' + num(audit.calibration.calibratedBrier)],
    ['Brier parts', 'rel ' + num(audit.calibration.testDecomposition.reliability) +
      ' - res ' + num(audit.calibration.testDecomposition.resolution) +
      ' + unc ' + num(audit.calibration.testDecomposition.uncertainty)],
    ['PPV gap', pct(audit.baseline.metrics.groups.ppvDisparity)],
    ['NPV gap', pct(audit.baseline.metrics.groups.npvDisparity)],
    ['FPR gap', pct(audit.baseline.metrics.groups.fprDisparity)],
    ['engine time', Math.round(audit.performance.totalMs) + ' ms'],
    ['end to end', Math.round(wallMs) + ' ms']
  ];
  for (const [label, value] of items) {
    const span = document.createElement('span');
    span.className = 'mini-stat';
    span.append(document.createTextNode(label + ' '));
    const strong = document.createElement('strong');
    strong.textContent = String(value);
    span.append(strong);
    wrap.append(span);
  }
}

function renderCounterexamples(audit) {
  const wrap = $('counterexamples');
  const countEl = $('counterexample-count');
  wrap.replaceChildren();
  countEl.textContent = audit.counterexamples.length > 9
    ? 'Showing 9 of ' + audit.counterexamples.length + ' counterexamples'
    : '';
  if (!audit.counterexamples.length) {
    const p = document.createElement('p');
    p.className = 'empty-state';
    p.textContent = 'No held-out case meets the counterexample rule at this setting. Try raising label bias or lowering the threshold.';
    wrap.append(p);
    return;
  }
  for (const c of audit.counterexamples.slice(0, 9)) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'case-card';
    card.setAttribute('aria-expanded', 'false');
    // Concise accessible name; the full detail text lives in the expanded
    // body so the button's name does not read as a paragraph.
    card.setAttribute('aria-label',
      'Held-out case ' + c.id + ', ' + c.groupLabel + ', ' + pct(c.score) + ' support score, missed globally');

    // A button may only hold phrasing content: every inner element is a span
    // styled as a block, keeping the markup valid for assistive tech.
    const head = document.createElement('span');
    head.className = 'case-head';
    const idSpan = document.createElement('span');
    idSpan.textContent = 'Held-out case #' + c.id;
    const badge = document.createElement('span');
    badge.className = 'case-badge';
    badge.textContent = 'missed globally';
    head.append(idSpan, badge);

    const title = document.createElement('span');
    title.className = 'case-title';
    title.textContent = pct(c.score) + ' support score';
    const sub = document.createElement('span');
    sub.className = 'case-sub';
    sub.textContent = c.groupLabel + ' | global threshold predicted no support';
    const flips = document.createElement('span');
    flips.className = 'case-flips';
    for (const flip of c.flips) {
      const tag = document.createElement('span');
      tag.className = 'case-flip';
      tag.textContent = flip.label;
      flips.append(tag);
    }
    const detail = document.createElement('span');
    detail.className = 'case-detail';
    for (const [key, value] of Object.entries(c.features)) {
      const row = document.createElement('span');
      row.className = 'case-feature';
      const k = document.createElement('span');
      k.textContent = key;
      const v = document.createElement('strong');
      v.textContent = String(value);
      row.append(k, v);
      detail.append(row);
    }
    const explain = document.createElement('span');
    explain.className = 'case-sub case-explain';
    explain.textContent = 'True label: needs support. Mitigation threshold: ' +
      c.flips.map((f) => f.label + ' ' + f.threshold.toFixed(2)).join(', ') + '.';
    detail.append(explain);

    card.append(head, title, sub, flips, detail);
    card.addEventListener('click', () => {
      const open = card.getAttribute('aria-expanded') === 'true';
      card.setAttribute('aria-expanded', open ? 'false' : 'true');
    });
    wrap.append(card);
  }
}

function renderEthics(audit) {
  $('ethics-benefits').textContent = audit.ethics.benefits;
  $('ethics-harms').textContent = audit.ethics.harms;
  $('ethics-failures').textContent = audit.ethics.failures;
  $('ethics-human').textContent = audit.ethics.human;
}

function renderHidden(audit) {
  const b = audit.baseline.metrics;
  const eo = audit.strategies.find((s) => s.id === 'equal-opportunity');
  const g = b.groups;
  const harmedIsB = g.groupB.fnr >= g.groupA.fnr;
  const harmed = harmedIsB ? g.groupB : g.groupA;
  const other = harmedIsB ? g.groupA : g.groupB;
  const harmedLabel = audit.groupLabels[harmedIsB ? 1 : 0] || 'Harmed group';
  const otherLabel = audit.groupLabels[harmedIsB ? 0 : 1] || 'Other group';
  const gapBefore = g.fnrDisparity;
  const gapAfter = eo.metrics.groups.fnrDisparity;
  const recall = eo.metrics.recall - b.recall;
  $('h-accuracy').textContent = pct(b.accuracy);
  $('h-auc').textContent = num(b.auc);
  $('h-fnr').textContent = pct(harmed.fnr);
  $('h-fnr-label').textContent = harmedLabel + ' missed rate';
  $('h-fnr0').textContent = pct(other.fnr);
  $('h-fnr0-label').textContent = otherLabel + ' missed rate';
  $('h-disp').textContent = pct(gapBefore) + ' → ' + pct(gapAfter);
  $('h-recall').textContent = (recall >= 0 ? '+' : '') + (recall * 100).toFixed(1) + ' pp';
  $('hidden-headline').textContent =
    pct(b.accuracy) + ' overall accuracy hides a ' + pct(harmed.fnr) + ' missed rate for ' + harmedLabel +
    '. Equal opportunity cuts the gap from ' + pct(gapBefore) + ' to ' + pct(gapAfter) +
    ' at ' + (recall >= 0 ? '+' : '') + (recall * 100).toFixed(1) + ' points of recall.';
  const ci = harmed.fnrCi;
  const ciText = ci
    ? ' ' + harmedLabel + ' missed-rate 95% CI ' + pct(ci.lo) + ' to ' + pct(ci.hi) + ' (n=' + harmed.n + ').'
    : '';
  $('hidden-note').textContent =
    'Same held-out rows: ' + num(b.auc) + ' ROC-AUC, ' + pct(other.fnr) + ' missed rate for ' + otherLabel +
    '.' + ciText + ' Thresholds were fitted on training rows only; no tested row influenced the fix.';
}

function renderCharts(audit) {
  drawRoc($('roc-canvas'), audit.roc, audit.models.lr.auc);
  drawReliability($('reliability-canvas'), audit.reliability);
  drawDisparity($('disparity-canvas'), audit.strategies, audit.groupLabels);
  drawLoss($('loss-canvas'), audit.models.lr.loss);
  $('roc-sub').textContent = audit.testCount + ' held-out rows, AUC ' + num(audit.models.lr.auc);
  $('reliability-sub').textContent = '10 bins, ECE ' + num(audit.baseline.metrics.ece);
  // Text equivalent of the disparity chart's tooltip-only numbers, kept in
  // sync with every render so keyboard and screen-reader users get the same
  // per-group rates without hovering.
  const parts = [];
  for (const s of audit.strategies) {
    const g = s.metrics.groups;
    const a = g.groups && g.groups.length ? g.groups[0] : null;
    const b = g.groups && g.groups.length > 1 ? g.groups[g.groups.length - 1] : a;
    if (a === null || b === null || !g.groupA) continue;
    const la = (audit.groupLabels && audit.groupLabels[a]) || 'group ' + a;
    const lb = (audit.groupLabels && audit.groupLabels[b]) || 'group ' + b;
    parts.push(s.label + ': ' + la + ' TPR ' + pct(g.groupA.tpr) + ' FNR ' + pct(g.groupA.fnr) +
      '; ' + lb + ' TPR ' + pct(g.groupB.tpr) + ' FNR ' + pct(g.groupB.fnr) + '.');
  }
  $('disparity-canvas').setAttribute('aria-label',
    'Bar chart of true positive and false negative rates by subgroup and policy. ' + parts.join(' '));
}

/* ------------------------------------------------------------------ */
/* Calibration workbench                                               */
/* ------------------------------------------------------------------ */

function ciText(ci, format) {
  const f = format || pct;
  return ci ? ' (' + f(ci.lo) + ' to ' + f(ci.hi) + ')' : '';
}

function renderWorkbench(audit) {
  const wb = audit.workbench;
  if (!wb) return;
  const u = wb.uncertainty;
  $('relwb-sub').textContent =
    'ECE ' + num(u.ece) + ', MCE ' + num(u.mce) + ', Brier ' + num(u.brier);
  $('frontier-sub').textContent = wb.frontier.points.length + ' achievable operating points';
  drawReliabilitySvg($('reliability-svg'), {
    bins: wb.bins,
    overlay: wb.groupOverlay,
    groupLabels: audit.groupLabels,
    showOverlay: $('overlay-groups').checked,
    onHover: (text) => {
      $('relwb-caption').textContent = text ||
        'ECE ' + num(u.ece) + ' (weighted mean bin gap), worst bin ' + num(u.mce) +
        '. Stems show the gap to perfect calibration; whiskers are Wilson 95% intervals.';
    }
  });
  drawFrontierSvg($('frontier-svg'), {
    points: wb.frontier.points,
    byGroup: wb.frontier.byGroup,
    groupLabels: audit.groupLabels,
    policyPoints: wb.frontier.policyPoints,
    onHover: (text) => {
      $('frontier-caption').textContent = text ||
        'Each point is a threshold: alerts per 100 held-out records vs missed positives per 100.';
    }
  });
  const legend = $('frontier-legend');
  legend.replaceChildren();
  for (const item of frontierLegend(wb.frontier.policyPoints, audit.groupLabels)) {
    const li = document.createElement('li');
    const sw = document.createElement('span');
    sw.className = 'swatch' + (item.marker ? ' marker' : '');
    if (item.dash) sw.style.borderTop = '2px dashed ' + item.color;
    else sw.style.background = item.color;
    li.append(sw, document.createTextNode(item.label));
    legend.append(li);
  }
  renderComparator(audit);
  renderSlices(audit);
}

function signedDelta(v, isCount) {
  if (isCount) return (v > 0 ? '+' : '') + v;
  return (v > 0 ? '+' : '') + (v * 100).toFixed(1) + ' pp';
}

function deltaClass(v) {
  if (v > 0.0005 || v >= 1) return 'delta-worse';
  if (v < -0.0005 || v <= -1) return 'delta-better';
  return 'delta-flat';
}

function renderComparator(audit) {
  const tbody = $('comparator-rows');
  tbody.replaceChildren();
  const comp = audit.workbench.comparator;
  for (const p of comp.policies) {
    const tr = document.createElement('tr');
    const nameCell = document.createElement('td');
    nameCell.className = 'policy-name';
    nameCell.append(document.createTextNode(p.label));
    const small = document.createElement('small');
    small.textContent = p.blurb + ' Fitted: ' + p.fittedNote + '.';
    nameCell.append(small);
    tr.append(nameCell);

    const thrText = typeof p.threshold === 'number'
      ? p.threshold.toFixed(3)
      : Object.entries(p.threshold)
          .map(([g, t]) => (audit.groupLabels[g] || 'G' + g) + ' ' + Number(t).toFixed(3))
          .join(' | ');
    const thrCell = document.createElement('td');
    thrCell.className = 'policy-thr';
    thrCell.textContent = thrText;
    tr.append(thrCell);

    const cells = [
      { text: pct(p.eval.recall), cls: '' },
      { text: signedDelta(p.deltas.missed, true), cls: deltaClass(p.deltas.missed) },
      { text: signedDelta(p.deltas.alerts, true), cls: deltaClass(p.deltas.alerts) },
      { text: pct(p.eval.gaps.equalOpportunity), cls: '' },
      { text: pct(p.eval.gaps.fnr), cls: '' },
      { text: pct(p.eval.gaps.ppv), cls: '' },
      { text: num(p.eval.decisionGap), cls: '' }
    ];
    for (const c of cells) {
      const td = document.createElement('td');
      td.textContent = c.text;
      if (c.cls) td.className = c.cls;
      tr.append(td);
    }
    tbody.append(tr);
  }
  const floorP = comp.policies.find((p) => p.id === 'sensitivity-floor');
  const noteParts = [
    'Deltas are exact held-out count changes vs the single global threshold: missed is false negatives, alerts is every flagged record.'
  ];
  if (floorP) {
    noteParts.push(
      'The ' + Math.round(comp.floor * 100) + '% sensitivity floor ' +
      (floorP.deltas.missed <= 0 ? 'recovers ' + (-floorP.deltas.missed) + ' missed case' + (-floorP.deltas.missed === 1 ? '' : 's') : 'misses ' + floorP.deltas.missed + ' more') +
      ' at the cost of ' + (floorP.deltas.alerts >= 0 ? '+' : '') + floorP.deltas.alerts + ' alerts held out.'
    );
  }
  $('comparator-note').textContent = noteParts.join(' ');
}

function renderSlices(audit) {
  const tbody = $('slices-rows');
  tbody.replaceChildren();
  const slices = audit.workbench.slices;
  for (const c of slices.cells) {
    const tr = document.createElement('tr');
    if (c.smallCell) tr.className = 'smallcell';
    const cells = [
      { text: c.label, cls: 'policy-name' },
      { text: String(c.n), cls: '' },
      { text: pct(c.fnr) + ciText(c.fnrCi), cls: '' },
      { text: pct(c.selectionRate), cls: '' },
      { text: pct(c.ppv) + ciText(c.ppvCi), cls: '' },
      { text: c.smallCell ? 'small cell' : 'ok', cls: c.smallCell ? 'flag-warn' : 'flag-ok' }
    ];
    for (const cd of cells) {
      const td = document.createElement('td');
      td.textContent = cd.text;
      if (cd.cls) td.className = cd.cls;
      tr.append(td);
    }
    tbody.append(tr);
  }
  $('slices-note').textContent = slices.warnings.length
    ? 'Small-cell warnings: ' + slices.warnings.join('; ') + '. The second axis is ' +
      (slices.second ? slices.second.dimension : 'a scenario feature') + ', cut at a fixed stated point.'
    : 'Every intersectional cell has n >= ' + slices.minN + '. The second axis is ' +
      (slices.second ? slices.second.dimension : 'a scenario feature') + ', cut at a fixed stated point.';
}

/* ------------------------------------------------------------------ */
/* Certificate, self tests and bookmarking                             */
/* ------------------------------------------------------------------ */

function renderPrintSummary(audit) {
  const b = audit.baseline.metrics;
  const eo = audit.strategies.find((s) => s.id === 'equal-opportunity').metrics;
  $('print-scenario').textContent = audit.scenarioName;
  $('print-params').textContent = 'bias ' + Number($('bias').value).toFixed(2) + ' | threshold ' +
    Number($('threshold').value).toFixed(2) + ' | seed ' + $('seed').value;
  $('print-rows').textContent = audit.trainCount + ' train / ' + audit.testCount + ' held out';
  $('print-accuracy').textContent = pct(b.accuracy);
  $('print-auc').textContent = num(b.auc);
  $('print-brier').textContent = num(b.brier);
  $('print-gap').textContent = pct(b.groups.equalOpportunityGap) + ' -> ' + pct(eo.groups.equalOpportunityGap);
  $('print-recall').textContent = pct(b.recall) + ' -> ' + pct(eo.recall);
  $('print-digest').textContent = state.cert.sha256.slice(0, 16) + '...';
}

function renderCertificate() {
  if (!state.cert) return;
  $('cert-hash').textContent = state.cert.sha256;
  $('cert-sub').textContent =
    state.cert.input.scenario + ', seed ' + state.cert.input.seed + ', bias ' + state.cert.input.bias +
    ', threshold ' + state.cert.input.threshold + ', ' + state.cert.evidence.heldOutRows + ' held-out rows. Recomputed in this page.';
}

function updateHash(params) {
  try {
    const frag = encodeAuditParams(params);
    if (location.hash !== '#' + frag) history.replaceState(null, '', '#' + frag);
  } catch {
    // History API can be blocked in odd embeds; the audit still runs.
  }
}

function applyHash() {
  const decoded = decodeAuditParams(location.hash || '');
  if (!decoded) return false;
  $('scenario').value = decoded.scenario;
  $('bias').value = String(decoded.bias);
  $('threshold').value = String(decoded.threshold);
  $('seed').value = String(decoded.seed);
  return true;
}

function download(name, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.hidden = false;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.hidden = true; }, 2600);
}

function wire() {
  $('demo-hidden').addEventListener('click', () => {
    $('scenario').value = 'scholarship';
    $('bias').value = '0.85';
    $('threshold').value = '0.45';
    $('seed').value = '20270101';
    run();
    const el = document.getElementById('hidden-failure');
    if (el) {
      const top = window.scrollY + el.getBoundingClientRect().top - 12;
      window.scrollTo({ top: Math.max(0, top), behavior: 'auto' });
    }
  });
  $('scenario').addEventListener('change', run);
  for (const id of ['bias', 'threshold']) $(id).addEventListener('input', scheduleAudit);
  $('seed').addEventListener('change', run);
  $('theme-toggle').addEventListener('click', () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
    $('theme-toggle').setAttribute('aria-pressed', dark ? 'false' : 'true');
    if (state.audit) {
      renderCharts(state.audit);
      renderWorkbench(state.audit);
    }
  });
  $('export-json').addEventListener('click', () => {
    if (!state.audit) return;
    download('triage-lens-audit.json', JSON.stringify(state.audit, null, 2), 'application/json');
    toast('Audit JSON exported locally.');
  });
  $('export-csv').addEventListener('click', () => {
    if (!state.audit) return;
    // The JSON export carries audit.disclaimer; the CSV gets the same
    // synthetic-data statement as a leading comment row.
    download('triage-lens-strategies.csv', '# ' + DATA_STATEMENT.replace(/\s+/g, ' ') + '\r\n' + toCsv(exportAudit(state.audit)), 'text/csv');
    toast('Strategy CSV exported locally.');
  });
  $('overlay-groups').addEventListener('change', () => { if (state.audit) renderWorkbench(state.audit); });
  $('cert-json').addEventListener('click', () => {
    if (!state.cert) return;
    download('triage-lens-certificate.json', certificateJson(state.cert), 'application/json');
    toast('Certificate JSON exported locally.');
  });
  $('cert-csv').addEventListener('click', () => {
    if (!state.cert) return;
    download('triage-lens-certificate.csv', certificateCsv(state.cert), 'text/csv');
    toast('Certificate CSV exported locally.');
  });
  $('cert-md').addEventListener('click', () => {
    if (!state.cert) return;
    download('triage-lens-certificate.md', certificateMarkdown(state.cert), 'text/markdown');
    toast('Markdown report exported locally.');
  });
  $('copy-link').addEventListener('click', async () => {
    // Build the link from the inputs the current run actually used, not
    // location.href (in-page anchors can replace the audit hash) and not
    // the raw fields (an unclamped seed would encode a different audit).
    const frag = encodeAuditParams(state.cert.input);
    const url = location.href.split('#')[0] + '#' + frag;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(url);
        toast('Share link copied.');
      } else {
        toast('Copy the URL from the address bar to share this audit.');
      }
    } catch {
      toast('Copy the URL from the address bar to share this audit.');
    }
  });
  $('run-selftests').addEventListener('click', () => {
    const t0 = performance.now();
    const checks = runSelfTests(state.audit);
    const ms = Math.round(performance.now() - t0);
    const list = $('selftest-list');
    list.replaceChildren();
    let passed = 0;
    for (const c of checks) {
      const li = document.createElement('li');
      const mark = document.createElement('span');
      mark.className = 'st-mark ' + (c.pass ? 'st-pass' : 'st-fail');
      mark.textContent = c.pass ? '✓' : '✗';
      const name = document.createElement('span');
      name.textContent = c.name;
      const detail = document.createElement('span');
      detail.className = 'st-detail';
      detail.textContent = c.detail;
      li.append(mark, name, detail);
      list.append(li);
      if (c.pass) passed++;
    }
    $('selftest-summary').textContent =
      passed + ' of ' + checks.length + ' checks passed in ' + ms + ' ms, inside this page.';
  });
  window.addEventListener('hashchange', () => {
    if (applyHash()) run();
  });
  window.addEventListener('resize', () => { if (state.audit) renderCharts(state.audit); });
}

wire();
applyHash();
run();
