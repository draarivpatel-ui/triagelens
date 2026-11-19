/* TriageLens UI controller. ES module, zero dependencies, no network access. */
import { runAudit, exportAudit, SCENARIOS } from './ml-engine.js';
import { drawRoc, drawReliability, drawDisparity, drawLoss } from './charts.js';

const $ = (id) => document.getElementById(id);
const state = { audit: null, timer: null };
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
  const seed = Math.max(1, Math.floor(Number($('seed').value) || 1));

  $('bias-value').textContent = bias.toFixed(2);
  $('threshold-value').textContent = threshold.toFixed(2);
  $('seed-value').textContent = String(seed);
  $('scenario-note').textContent = scenarioNote(scenarioId);

  const t0 = performance.now();
  const audit = runAudit({ scenarioId, bias, threshold, seed });
  const wallMs = performance.now() - t0;
  state.audit = audit;

  renderHero(audit, wallMs);
  renderMetrics(audit);
  renderStrategies(audit);
  renderWeights(audit);
  renderTree(audit);
  renderMiniStats(audit, wallMs);
  renderCounterexamples(audit);
  renderEthics(audit);
  renderCharts(audit);
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
  $('m-gap-foot').textContent = drop > 0.0005
    ? 'reduced by ' + pct(drop) + ' with equal opportunity'
    : 'no gap at this setting';
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
    pct(eo.groups.equalOpportunityGap) + ' (a ' + pct(gapDrop) + ' reduction) for an accuracy change of ' +
    (accuracyCost >= 0 ? '-' : '+') + pct(Math.abs(accuracyCost)) + '. Group-aware thresholds are fitted on the training split only.';
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
  wrap.replaceChildren();
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

    const head = document.createElement('div');
    head.className = 'case-head';
    const idSpan = document.createElement('span');
    idSpan.textContent = 'Held-out case #' + c.id;
    const badge = document.createElement('span');
    badge.className = 'case-badge';
    badge.textContent = 'missed globally';
    head.append(idSpan, badge);

    const title = document.createElement('p');
    title.className = 'case-title';
    title.textContent = pct(c.score) + ' support score';
    const sub = document.createElement('p');
    sub.className = 'case-sub';
    sub.textContent = c.groupLabel + ' | global threshold predicted no support';
    const flips = document.createElement('div');
    flips.className = 'case-flips';
    for (const flip of c.flips) {
      const tag = document.createElement('span');
      tag.className = 'case-flip';
      tag.textContent = flip.label;
      flips.append(tag);
    }
    const detail = document.createElement('div');
    detail.className = 'case-detail';
    for (const [key, value] of Object.entries(c.features)) {
      const row = document.createElement('div');
      row.className = 'case-feature';
      const k = document.createElement('span');
      k.textContent = key;
      const v = document.createElement('strong');
      v.textContent = String(value);
      row.append(k, v);
      detail.append(row);
    }
    const explain = document.createElement('p');
    explain.className = 'case-sub';
    explain.style.marginTop = '10px';
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

function renderCharts(audit) {
  drawRoc($('roc-canvas'), audit.roc, audit.models.lr.auc);
  drawReliability($('reliability-canvas'), audit.reliability);
  drawDisparity($('disparity-canvas'), audit.strategies, audit.groupLabels);
  drawLoss($('loss-canvas'), audit.models.lr.loss);
  $('roc-sub').textContent = audit.testCount + ' held-out rows, AUC ' + num(audit.models.lr.auc);
  $('reliability-sub').textContent = '10 bins, ECE ' + num(audit.baseline.metrics.ece);
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
  $('scenario').addEventListener('change', run);
  for (const id of ['bias', 'threshold']) $(id).addEventListener('input', scheduleAudit);
  $('seed').addEventListener('change', run);
  $('theme-toggle').addEventListener('click', () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
    if (state.audit) renderCharts(state.audit);
  });
  $('export-json').addEventListener('click', () => {
    if (!state.audit) return;
    download('triage-lens-audit.json', JSON.stringify(state.audit, null, 2), 'application/json');
    toast('Audit JSON exported locally.');
  });
  $('export-csv').addEventListener('click', () => {
    if (!state.audit) return;
    const rows = exportAudit(state.audit);
    const cols = Object.keys(rows[0]);
    const csv = [cols.join(',')].concat(rows.map((r) => cols.map((c) => JSON.stringify(r[c])).join(','))).join('\n');
    download('triage-lens-strategies.csv', csv, 'text/csv');
    toast('Strategy CSV exported locally.');
  });
  window.addEventListener('resize', () => { if (state.audit) renderCharts(state.audit); });
}

wire();
run();
