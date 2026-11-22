/* TriageLens audit certificate, URL bookmarking and self tests.
 * Pure ES module, zero dependencies, fully deterministic.
 *
 * A certificate is the canonical JSON of the audit inputs and outputs
 * plus a SHA-256 integrity digest computed in pure JavaScript
 * (FIPS 180-4). It is an integrity hash, not a signature: nothing here
 * claims a private key or a third-party attestation, and the page says
 * so honestly.
 */

import {
  SCENARIOS,
  DATA_STATEMENT,
  exportAudit,
  toCsv,
  parseCsv,
  expectedCalibrationError,
  brierDecomposition,
  wilsonInterval,
  runAudit
} from './ml-engine.js';
import {
  maximumCalibrationError,
  fitSensitivityFloor,
  sliceMetrics,
  thresholdFrontier,
  SMALL_CELL_N
} from './calibration.js';

/* ------------------------------------------------------------------ */
/* UTF-8 and base64url primitives (no external calls, no dependencies)  */
/* ------------------------------------------------------------------ */

function utf8Bytes(str) {
  if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(str);
  const s = String(str);
  const out = [];
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
    else if (c >= 0xd800 && c <= 0xdbff) {
      const c2 = s.charCodeAt(++i);
      const cp = 0x10000 + ((c & 0x3ff) << 10) + (c2 & 0x3ff);
      out.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
    } else out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  return new Uint8Array(out);
}

function utf8Decode(bytes) {
  if (typeof TextDecoder !== 'undefined') return new TextDecoder().decode(bytes);
  let s = '';
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    if (b < 0x80) s += String.fromCharCode(b);
    else if (b < 0xe0) s += String.fromCharCode(((b & 31) << 6) | (bytes[++i] & 63));
    else if (b < 0xf0) s += String.fromCharCode(((b & 15) << 12) | ((bytes[++i] & 63) << 6) | (bytes[++i] & 63));
    else {
      const cp = ((b & 7) << 18) | ((bytes[++i] & 63) << 12) | ((bytes[++i] & 63) << 6) | (bytes[++i] & 63);
      s += String.fromCharCode(0xd800 + ((cp - 0x10000) >> 10), 0xdc00 + ((cp - 0x10000) & 0x3ff));
    }
  }
  return s;
}

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

export function base64urlEncode(str) {
  const b = utf8Bytes(str);
  let out = '';
  for (let i = 0; i < b.length; i += 3) {
    const a = b[i];
    const c = i + 1 < b.length ? b[i + 1] : null;
    const d = i + 2 < b.length ? b[i + 2] : null;
    out += B64[a >> 2];
    out += B64[((a & 3) << 4) | (c === null ? 0 : c >> 4)];
    out += c === null ? '' : B64[((c & 15) << 2) | (d === null ? 0 : d >> 6)];
    out += d === null ? '' : B64[d & 63];
  }
  return out;
}

export function base64urlDecode(str) {
  const s = String(str);
  if (!/^[A-Za-z0-9_-]*$/.test(s) || s.length % 4 === 1) return null;
  const idx = (ch) => B64.indexOf(ch);
  const out = [];
  for (let i = 0; i < s.length; i += 4) {
    const c0 = idx(s[i]);
    const c1 = i + 1 < s.length ? idx(s[i + 1]) : 0;
    const c2 = i + 2 < s.length ? idx(s[i + 2]) : 64;
    const c3 = i + 3 < s.length ? idx(s[i + 3]) : 64;
    out.push((c0 << 2) | (c1 >> 4));
    if (c2 !== 64) out.push(((c1 & 15) << 4) | (c2 >> 2));
    if (c3 !== 64) out.push(((c2 & 3) << 6) | c3);
  }
  return utf8Decode(new Uint8Array(out));
}

/* ------------------------------------------------------------------ */
/* SHA-256 (FIPS 180-4)                                                 */
/* ------------------------------------------------------------------ */

const K256 = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
];

function rotr(x, n) {
  return (x >>> n) | (x << (32 - n));
}

/** SHA-256 hex digest of a string (UTF-8 encoded). FIPS 180-4. */
export function sha256Hex(message) {
  const bytes = utf8Bytes(String(message));
  const bitLenHi = Math.floor((bytes.length / 0x20000000)) >>> 0;
  const bitLenLo = (bytes.length << 3) >>> 0;
  const padded = new Uint8Array(((bytes.length + 9 + 63) >> 6 << 6));
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const dv = new DataView(padded.buffer);
  dv.setUint32(padded.length - 8, bitLenHi);
  dv.setUint32(padded.length - 4, bitLenLo);

  const h = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const w = new Int32Array(64);
  for (let off = 0; off < padded.length; off += 64) {
    for (let t = 0; t < 16; t++) w[t] = dv.getUint32(off + t * 4);
    for (let t = 16; t < 64; t++) {
      const s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >>> 3);
      const s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >>> 10);
      w[t] = (w[t - 16] + s0 + w[t - 7] + s1) | 0;
    }
    let [a, b, c, d, e, f, g, hh] = h;
    for (let t = 0; t < 64; t++) {
      const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + s1 + ch + K256[t] + w[t]) | 0;
      const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (s0 + maj) | 0;
      hh = g; g = f; f = e; e = (d + t1) | 0;
      d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    h[0] = (h[0] + a) | 0; h[1] = (h[1] + b) | 0; h[2] = (h[2] + c) | 0; h[3] = (h[3] + d) | 0;
    h[4] = (h[4] + e) | 0; h[5] = (h[5] + f) | 0; h[6] = (h[6] + g) | 0; h[7] = (h[7] + hh) | 0;
  }
  return h.map((x) => (x >>> 0).toString(16).padStart(8, '0')).join('');
}

/* ------------------------------------------------------------------ */
/* Canonical serialization                                             */
/* ------------------------------------------------------------------ */

/**
 * Deterministic JSON: object keys sorted at every depth, array order
 * preserved, non-finite numbers become null (matching JSON.stringify),
 * cycles fail closed with a clear error instead of hanging.
 */
export function canonicalize(value, _seen) {
  const seen = _seen || new Set();
  const canon = (v) => {
    if (v === null) return null;
    const t = typeof v;
    if (t === 'number') return Number.isFinite(v) ? v : null;
    if (t === 'boolean' || t === 'string') return v;
    if (t !== 'object') return undefined;
    if (seen.has(v)) throw new Error('cannot canonicalize a cyclic structure');
    seen.add(v);
    if (Array.isArray(v)) {
      const out = v.map((x) => {
        const c = canon(x);
        return c === undefined ? null : c;
      });
      seen.delete(v);
      return out;
    }
    const out = Object.create(null);
    for (const k of Object.keys(v).sort()) {
      const c = canon(v[k]);
      if (c !== undefined) out[k] = c;
    }
    seen.delete(v);
    return out;
  };
  return JSON.stringify(canon(value));
}

/* ------------------------------------------------------------------ */
/* The audit certificate                                               */
/* ------------------------------------------------------------------ */

export const CERT_FORMAT = 'triagelens-audit-certificate';
export const CERT_VERSION = 1;

/**
 * The hashed payload of a certificate. Deterministic: no wall clock, no
 * environment data, no random values. The same audit always produces
 * the same digest, so a judge can recompute it and compare.
 */
export function certificatePayload(audit) {
  const wb = (audit && audit.workbench) || {};
  const comp = (wb.comparator && wb.comparator.policies) || [];
  const cells = (wb.slices && wb.slices.cells) || [];
  const strategies = Array.isArray(audit && audit.strategies) ? audit.strategies : [];
  return {
    format: CERT_FORMAT,
    version: CERT_VERSION,
    generator: 'TriageLens local audit engine',
    syntheticDataOnly: true,
    disclaimer: (audit && audit.disclaimer) || DATA_STATEMENT,
    input: {
      scenario: audit ? audit.scenario : null,
      seed: audit ? audit.seed : null,
      bias: audit ? audit.bias : null,
      n: audit ? audit.n : null,
      trainFraction: audit ? audit.trainFraction : null,
      threshold: audit ? audit.threshold : null
    },
    evidence: {
      trainRows: audit ? audit.trainCount : null,
      heldOutRows: audit ? audit.testCount : null,
      // Binds the certificate to the actual predictions: the ROC and
      // reliability curves and every strategy's decision vector.
      scoreDigest: sha256Hex(canonicalize({
        roc: audit ? audit.roc : null,
        reliability: audit ? audit.reliability : null,
        predictions: strategies.map((s) => s.predictions || [])
      }))
    },
    results: {
      strategies: audit ? exportAudit(audit) : [],
      comparator: comp.map((p) => ({
        id: p.id,
        threshold: p.threshold,
        recall: p.eval.recall,
        missed: p.eval.counts.fn,
        alerts: p.eval.counts.tp + p.eval.counts.fp,
        eoGap: p.eval.gaps.equalOpportunity,
        fnrGap: p.eval.gaps.fnr,
        ppvGap: p.eval.gaps.ppv,
        decisionGap: p.eval.decisionGap
      })),
      uncertainty: wb.uncertainty || null,
      intersectional: {
        minN: (wb.slices && wb.slices.minN) || SMALL_CELL_N,
        cells: cells.map((c) => ({
          id: c.id, n: c.n, tpr: c.tpr, fnr: c.fnr, fpr: c.fpr, ppv: c.ppv, smallCell: c.smallCell
        }))
      }
    }
  };
}

/** Full certificate: payload plus its SHA-256 integrity digest. */
export function buildCertificate(audit) {
  const payload = certificatePayload(audit);
  return { ...payload, sha256: sha256Hex(canonicalize(payload)) };
}

/** Certificate JSON download text. */
export function certificateJson(cert) {
  return JSON.stringify(cert, null, 2) + '\n';
}

/**
 * Certificate CSV: the synthetic-data statement and digest ride along as
 * comment rows, then the strategy table. Every cell passes through the
 * formula-safe escaper, so nothing can execute in a spreadsheet app.
 */
export function certificateCsv(cert) {
  const rows = Array.isArray(cert && cert.results && cert.results.strategies) ? cert.results.strategies : [];
  const head =
    '# ' + String(cert.disclaimer || DATA_STATEMENT).replace(/\s+/g, ' ') + '\r\n' +
    '# integrity sha256: ' + cert.sha256 + '\r\n';
  return head + toCsv(rows);
}

/** Certificate Markdown report for judges. */
export function certificateMarkdown(cert) {
  const lines = [];
  lines.push('# TriageLens audit certificate');
  lines.push('');
  lines.push('- Scenario: ' + cert.input.scenario);
  lines.push('- Seed: ' + cert.input.seed);
  lines.push('- Bias: ' + cert.input.bias);
  lines.push('- Rows: ' + cert.input.n + ' (train ' + cert.evidence.trainRows + ', held out ' + cert.evidence.heldOutRows + ')');
  lines.push('- Threshold: ' + cert.input.threshold);
  lines.push('');
  lines.push('| Policy | Threshold | Recall | Missed | Alerts | EO gap | FNR gap |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- |');
  for (const p of cert.results.comparator) {
    const th = typeof p.threshold === 'number'
      ? p.threshold.toFixed(3)
      : JSON.stringify(p.threshold);
    lines.push('| ' + p.id + ' | ' + th + ' | ' + p.recall.toFixed(3) + ' | ' +
      p.missed + ' | ' + p.alerts + ' | ' + (p.eoGap * 100).toFixed(1) + '% | ' +
      (p.fnrGap * 100).toFixed(1) + '% |');
  }
  lines.push('');
  const u = cert.results.uncertainty || {};
  lines.push('Held-out ECE ' + (u.ece ?? 0).toFixed(4) + ', MCE ' + (u.mce ?? 0).toFixed(4) +
    ', Brier ' + (u.brier ?? 0).toFixed(4) + '.');
  lines.push('');
  const inter = cert.results.intersectional || { minN: SMALL_CELL_N, cells: [] };
  const small = (inter.cells || []).filter((c) => c.smallCell);
  if (small.length) {
    lines.push('Small-cell caution: ' + small.length + ' of ' + inter.cells.length +
      ' intersectional cells fall below the n=' + inter.minN + ' reliability floor (' +
      small.map((c) => c.id).join(', ') + ').');
    lines.push('Rates on flagged cells have wide Wilson intervals; read them as directional only.');
    lines.push('');
  }
  lines.push('```');
  lines.push('sha256 ' + cert.sha256);
  lines.push('evidence ' + cert.evidence.scoreDigest);
  lines.push('```');
  lines.push('');
  lines.push('The digest is an integrity hash over the canonical JSON of the audit');
  lines.push('inputs and outputs, computed locally by this page.');
  lines.push('It is not a cryptographic signature and no private key exists.');
  lines.push('');
  lines.push(cert.disclaimer);
  lines.push('');
  return lines.join('\n');
}

/* ------------------------------------------------------------------ */
/* URL-hash bookmarking                                                */
/* ------------------------------------------------------------------ */

/**
 * Encode audit controls into a deterministic, URL-safe fragment.
 * Format: audit=<base64url(JSON {v,s,b,t,seed})> with no padding so the
 * fragment survives every link rewriter.
 */
export function encodeAuditParams(params) {
  const p = params || {};
  const obj = {
    v: 1,
    s: String(p.scenario ?? 'clinic'),
    // Missing fields fall back to the control defaults so every encoded
    // bookmark still decodes; the app always passes the live values.
    b: Number.isFinite(p.bias) ? Number(p.bias) : 0.6,
    t: Number.isFinite(p.threshold) ? Number(p.threshold) : 0.35,
    seed: Math.floor(Number(p.seed)) >>> 0
  };
  return 'audit=' + base64urlEncode(JSON.stringify(obj));
}

/**
 * Decode and strictly validate a bookmark fragment. Accepts
 * 'audit=...', '#audit=...', '#...' or the bare payload. Returns
 * {scenario, bias, threshold, seed} or null on anything malformed;
 * it never throws and never returns an out-of-range value.
 */
export function decodeAuditParams(fragment) {
  let s = String(fragment == null ? '' : fragment).trim();
  if (!s) return null;
  if (s.startsWith('#')) s = s.slice(1);
  if (s.startsWith('audit=')) s = s.slice(6);
  const text = base64urlDecode(s);
  if (text === null) return null;
  let obj;
  try {
    obj = JSON.parse(text);
  } catch {
    return null;
  }
  if (!obj || typeof obj !== 'object') return null;
  const ids = SCENARIOS.map((x) => x.id);
  const scenario = typeof obj.s === 'string' && ids.includes(obj.s) ? obj.s : null;
  const bias = Number(obj.b);
  const threshold = Number(obj.t);
  const seed = Number(obj.seed);
  if (scenario === null) return null;
  // Slider ranges, not the looser engine bounds: a bookmark must only
  // carry values the controls can actually express (bias 0..0.9,
  // threshold 0.1..0.8), or the link would reload to different numbers.
  if (!Number.isFinite(bias) || bias < 0 || bias > 0.9) return null;
  if (!Number.isFinite(threshold) || threshold < 0.1 || threshold > 0.8) return null;
  if (!Number.isFinite(seed) || seed < 1 || seed > 999999999 || Math.floor(seed) !== seed) return null;
  return { scenario, bias, threshold, seed };
}

/* ------------------------------------------------------------------ */
/* In-page self tests                                                  */
/* ------------------------------------------------------------------ */

const KAT = {
  empty: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
  abc: 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
  len56: '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
  million: 'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0'
};

function expect(cond, detail) {
  if (!cond) throw new Error(detail || 'check failed');
  return detail || 'ok';
}

/**
 * The same checks the page's self-test runner executes. Each entry is
 * {id, name, pass, detail}; the function never throws, so the UI can
 * always render a full pass/fail list. When `audit` is omitted a small
 * deterministic audit is run for the integration checks.
 */
export function runSelfTests(audit) {
  const checks = [];
  const push = (id, name, fn) => {
    try {
      checks.push({ id, name, pass: true, detail: fn() || 'ok' });
    } catch (e) {
      checks.push({ id, name, pass: false, detail: String((e && e.message) || e) });
    }
  };

  const kat = (id, msg, want) => push(id, 'SHA-256 ' + id.replace('sha256-', '') + ' known-answer vector', () => {
    const d = sha256Hex(msg);
    expect(d === want, d.slice(0, 16));
    return d.slice(0, 16);
  });
  kat('sha256-empty', '', KAT.empty);
  kat('sha256-abc', 'abc', KAT.abc);
  kat('sha256-56byte', 'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq', KAT.len56);
  kat('sha256-million', 'a'.repeat(1000000), KAT.million);
  push('canonical-order', 'canonical JSON is key-order invariant', () =>
    expect(
      sha256Hex(canonicalize({ b: 1, a: [3, { d: 2, c: 1 }] })) ===
      sha256Hex(canonicalize({ a: [3, { c: 1, d: 2 }], b: 1 })),
      'digests differ'));

  push('ece-bounds', 'ECE stays inside [0, 1] on every vector', () => {
    const vectors = [
      [[0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0]],
      [[0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]],
      [[0.9, 0.9, 0.9], [0, 0, 0]],
      [[0.5], [1]],
      [[], []]
    ];
    for (const [s, y] of vectors) {
      const e = expectedCalibrationError(s, y, 10);
      expect(Number.isFinite(e) && e >= 0 && e <= 1, 'ece=' + e);
    }
    return vectors.length + ' vectors';
  });
  push('mce-dominates-ece', 'MCE is never below ECE', () => {
    const vectors = [
      [[0.9, 0.1, 0.8, 0.2, 0.55, 0.45], [1, 0, 1, 0, 1, 0]],
      [[0.31, 0.32, 0.67, 0.68], [0, 1, 1, 0]],
      [[0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]]
    ];
    for (const [s, y] of vectors) {
      const ece = expectedCalibrationError(s, y, 10);
      const mce = maximumCalibrationError(s, y, 10);
      expect(mce >= ece - 1e-12, 'mce ' + mce + ' < ece ' + ece);
    }
    return vectors.length + ' vectors';
  });
  push('brier-identity', 'Brier score equals reliability minus resolution plus uncertainty', () => {
    const vectors = [
      [[0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1]],
      [[0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1]],
      [[0.37, 0.5, 0.5, 0.91], [1, 0, 1, 1]]
    ];
    for (const [s, y] of vectors) {
      const d = brierDecomposition(s, y);
      const err = Math.abs(d.brier - (d.reliability - d.resolution + d.uncertainty));
      expect(err < 1e-10, 'identity error ' + err);
    }
    return vectors.length + ' vectors';
  });
  push('wilson-ordering', 'Wilson interval always orders lo <= p <= hi', () => {
    for (const n of [1, 5, 30, 100, 1000]) {
      for (let k = 0; k <= n; k += Math.max(1, Math.floor(n / 7))) {
        const ci = wilsonInterval(k, n);
        const p = k / n;
        expect(ci.lo >= 0 && ci.hi <= 1 && ci.lo <= ci.hi, JSON.stringify(ci));
        expect(ci.lo <= p + 1e-9 && ci.hi >= p - 1e-9, 'CI misses ' + p);
      }
    }
    return 'grid ok';
  });
  push('wilson-shrink', 'Wilson interval width shrinks as n grows', () => {
    let prev = Infinity;
    for (const n of [10, 50, 200, 1000]) {
      const ci = wilsonInterval(Math.round(n / 2), n);
      const w = ci.hi - ci.lo;
      expect(w < prev + 1e-9, 'width did not shrink at n=' + n);
      prev = w;
    }
    return 'monotone';
  });
  push('threshold-monotone', 'alert volume falls and missed cases rise as the threshold climbs', () => {
    const scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    const ys = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1];
    const ids = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    const f = thresholdFrontier(scores, ys, ids);
    let prevSel = Infinity;
    let prevFnr = -Infinity;
    for (const pt of f.points) {
      expect(pt.sel <= prevSel + 1e-12, 'selection rose at t=' + pt.t);
      expect(pt.fnr >= prevFnr - 1e-12, 'FNR fell at t=' + pt.t);
      prevSel = pt.sel;
      prevFnr = pt.fnr;
    }
    return f.points.length + ' points';
  });
  push('small-cell-guard', 'small-cell warning fires exactly below n=' + SMALL_CELL_N, () => {
    const rows = [];
    const scores = [];
    const groupIds = [];
    const sliceIds = [];
    for (let i = 0; i < SMALL_CELL_N - 1; i++) { rows.push({ y: i % 2 }); scores.push(0.6); groupIds.push(0); sliceIds.push(0); }
    for (let i = 0; i < SMALL_CELL_N; i++) { rows.push({ y: i % 2 }); scores.push(0.6); groupIds.push(0); sliceIds.push(1); }
    const sm = sliceMetrics({ rows, scores, groupIds, sliceIds, threshold: 0.5 });
    const small = sm.cells.find((c) => c.id === '0|0');
    const ok = sm.cells.find((c) => c.id === '0|1');
    expect(small.smallCell === true && !!small.warning, 'n=29 cell not flagged');
    expect(ok.smallCell === false && ok.warning === null, 'n=30 cell wrongly flagged');
    return 'boundary exact';
  });
  push('floor-recall', 'sensitivity-floor guardrail meets its recall target on fitting rows', () => {
    const scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
    const ys = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0];
    const fit = fitSensitivityFloor(scores, ys, 0.9);
    expect(fit.achievable, 'floor marked unreachable');
    expect(fit.recall >= 0.9 - 1e-12, 'fitted recall ' + fit.recall);
    return 't=' + fit.threshold;
  });
  push('bookmark-roundtrip', 'audit URL bookmark encodes and decodes identically', () => {
    const p = { scenario: 'scholarship', bias: 0.85, threshold: 0.45, seed: 20270101 };
    const back = decodeAuditParams('#' + encodeAuditParams(p));
    expect(back && back.scenario === p.scenario && back.bias === p.bias &&
      back.threshold === p.threshold && back.seed === p.seed, JSON.stringify(back));
    expect(decodeAuditParams('audit=!!!!') === null, 'junk accepted');
    expect(decodeAuditParams('audit=' + base64urlEncode('not json')) === null, 'non-JSON accepted');
    expect(decodeAuditParams('audit=' + base64urlEncode('{"s":"nope","b":0.5,"t":0.5,"seed":1}')) === null, 'bad scenario accepted');
    expect(decodeAuditParams('audit=' + base64urlEncode('{"s":"clinic","b":7,"t":0.5,"seed":1}')) === null, 'bad bias accepted');
    return 'round-trip + rejects';
  });
  push('cert-deterministic', 'certificate digest is deterministic and input-bound', () => {
    const a = audit || runAudit({ seed: 7, n: 200, epochs: 60 });
    const c1 = buildCertificate(a);
    const c2 = buildCertificate(a);
    expect(c1.sha256 === c2.sha256, 'digest drifted');
    expect(/^[0-9a-f]{64}$/.test(c1.sha256), 'digest not hex');
    const c3 = buildCertificate(runAudit({ seed: 8, n: 200, epochs: 60 }));
    expect(c3.sha256 !== c1.sha256, 'different inputs gave the same digest');
    return c1.sha256.slice(0, 16);
  });
  push('csv-safe', 'certificate CSV carries no live formula cells', () => {
    const a = audit || runAudit({ seed: 7, n: 200, epochs: 60 });
    const csv = certificateCsv(buildCertificate(a));
    const body = csv.split(/\r?\n/).filter((l) => !l.startsWith('#')).join('\n');
    const parsed = parseCsv(body);
    let checked = 0;
    for (const row of parsed.rows) {
      for (const v of Object.values(row)) {
        const s = String(v);
        expect(!/^[=+\-@\t\r]/.test(s) || s.startsWith("'"), 'unsafe cell ' + s);
        checked++;
      }
    }
    return checked + ' cells';
  });
  return checks;
}
