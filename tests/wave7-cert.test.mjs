/* Wave 7 tests: audit certificate, SHA-256, URL bookmarking, self-tests.
 * The pure-JS SHA-256 is checked against node:crypto, the FIPS 180-4
 * reference implementation available to every Node runtime.
 * Run: node --test tests/wave7-cert.test.mjs */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { runAudit, DATA_STATEMENT, parseCsv } from '../src/ml-engine.js';
import {
  CERT_FORMAT,
  CERT_VERSION,
  sha256Hex,
  canonicalize,
  base64urlEncode,
  base64urlDecode,
  buildCertificate,
  certificateJson,
  certificateCsv,
  certificateMarkdown,
  encodeAuditParams,
  decodeAuditParams,
  runSelfTests
} from '../src/audit-cert.js';

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, '..');
const read = (p) => readFileSync(join(repo, p), 'utf8');
const refSha = (s) => createHash('sha256').update(s, 'utf8').digest('hex');

/* ------------------------------------------------------------------ */
/* SHA-256: FIPS 180-4 known-answer vectors                            */
/* ------------------------------------------------------------------ */

test('sha256Hex matches the empty-string FIPS vector', () => {
  assert.equal(sha256Hex(''), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
});

test('sha256Hex matches the abc FIPS vector', () => {
  assert.equal(sha256Hex('abc'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
});

test('sha256Hex matches the 56-byte multi-block FIPS vector', () => {
  assert.equal(
    sha256Hex('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'),
    '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1'
  );
});

test('sha256Hex matches the million-a FIPS vector', () => {
  assert.equal(
    sha256Hex('a'.repeat(1000000)),
    'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0'
  );
});

test('sha256Hex matches node:crypto across padding boundary lengths', () => {
  // Block boundary is 64 bytes; padding edge cases are 55, 56, 63, 64,
  // 119 and 127 byte messages plus one byte either side.
  for (const len of [1, 2, 54, 55, 56, 57, 62, 63, 64, 65, 118, 119, 120, 127, 128, 129, 255, 256]) {
    const s = 'x'.repeat(len);
    assert.equal(sha256Hex(s), refSha(s), 'length ' + len);
  }
});

test('sha256Hex matches node:crypto on multi-byte UTF-8 input', () => {
  for (const s of [
    'caf\u00e9',
    '\u4f60\u597d\u4e16\u754c',
    'risk \u00b1 0.05',
    '\ud83d\ude00 emoji \ud83c\udfaf',
    'mixed ascii and \u00e9\u00e8\u00ea long enough to cross a block boundary ' + '\u00fc'.repeat(80)
  ]) {
    assert.equal(sha256Hex(s), refSha(s), JSON.stringify(s).slice(0, 40));
  }
});

test('sha256Hex output is 64 lowercase hex and deterministic', () => {
  const h = sha256Hex('triagelens');
  assert.match(h, /^[0-9a-f]{64}$/);
  assert.equal(h, sha256Hex('triagelens'));
  assert.notEqual(sha256Hex('a'), sha256Hex('b'));
});

/* ------------------------------------------------------------------ */
/* utf8/base64url primitives                                           */
/* ------------------------------------------------------------------ */

test('base64urlEncode uses only the URL-safe alphabet', () => {
  const encoded = base64urlEncode('{"s":"clinic","b":0.6,"t":0.35,"seed":20270101}');
  assert.match(encoded, /^[A-Za-z0-9_-]+$/);
  assert.ok(!encoded.includes('='), 'no padding');
});

test('base64urlEncode matches Buffer base64url for varied payloads', () => {
  for (const s of ['', 'a', 'ab', 'abc', 'hello world', '{"x":[1,2,3]}', '\u00e9\u00e8', 'f'.repeat(97)]) {
    const expected = Buffer.from(s, 'utf8').toString('base64url');
    assert.equal(base64urlEncode(s), expected, JSON.stringify(s).slice(0, 30));
  }
});

test('base64urlDecode round-trips through encode', () => {
  for (const s of ['hello', '{"a":1}', '\u4f60\u597d', 'x'.repeat(200)]) {
    assert.equal(base64urlDecode(base64urlEncode(s)), s);
  }
});

test('base64urlDecode returns null on malformed input', () => {
  for (const bad of ['!!!', 'a b c', 'a', '====', 'a=b']) {
    assert.equal(base64urlDecode(bad), null, JSON.stringify(bad));
  }
  assert.equal(base64urlDecode(''), '');
});

/* ------------------------------------------------------------------ */
/* canonicalize                                                        */
/* ------------------------------------------------------------------ */

test('canonicalize sorts object keys recursively and keeps array order', () => {
  const a = canonicalize({ b: 1, a: [3, { d: 1, c: 2 }] });
  const b = canonicalize({ a: [3, { c: 2, d: 1 }], b: 1 });
  assert.equal(a, b);
  assert.equal(a, '{"a":[3,{"c":2,"d":1}],"b":1}');
  // Arrays are order-sensitive: never sorted.
  assert.notEqual(canonicalize({ a: [2, 1] }), canonicalize({ a: [1, 2] }));
});

test('canonicalize maps non-finite numbers to null like JSON.stringify', () => {
  assert.equal(canonicalize({ x: NaN }), '{"x":null}');
  assert.equal(canonicalize({ x: Infinity, y: -Infinity }), '{"x":null,"y":null}');
  assert.equal(canonicalize([NaN]), '[null]');
});

test('canonicalize drops undefined and functions exactly like JSON', () => {
  assert.equal(canonicalize({ a: 1, b: undefined }), '{"a":1}');
  assert.equal(canonicalize({ a: 1, f: () => 2 }), '{"a":1}');
  assert.equal(canonicalize([1, undefined, 3]), '[1,null,3]');
});

test('canonicalize preserves a __proto__ data key instead of swallowing it', () => {
  const obj = JSON.parse('{"__proto__": 7, "z": 1}');
  const out = canonicalize(obj);
  assert.ok(out.includes('"__proto__":7'), out);
  assert.equal(sha256Hex(out).length, 64);
});

test('canonicalize throws on cyclic input rather than looping forever', () => {
  const cyc = { a: 1 };
  cyc.self = cyc;
  assert.throws(() => canonicalize(cyc));
  const nested = { list: [] };
  nested.list.push(nested);
  assert.throws(() => canonicalize(nested));
});

test('canonicalize handles the primitives and nested escapes', () => {
  assert.equal(canonicalize('hi'), '"hi"');
  assert.equal(canonicalize(true), 'true');
  assert.equal(canonicalize(null), 'null');
  assert.equal(canonicalize({ s: 'a"b\nc' }), '{"s":"a\\"b\\nc"}');
});

/* ------------------------------------------------------------------ */
/* buildCertificate                                                    */
/* ------------------------------------------------------------------ */

function auditFixture(opts = {}) {
  return runAudit({ scenarioId: 'clinic', seed: 4242, n: 500, bias: 0.6, threshold: 0.4, ...opts });
}

test('buildCertificate exposes the format constants and required blocks', () => {
  const cert = buildCertificate(auditFixture());
  assert.equal(cert.format, CERT_FORMAT);
  assert.equal(cert.version, CERT_VERSION);
  for (const k of ['generator', 'syntheticDataOnly', 'disclaimer', 'input', 'evidence', 'results']) {
    assert.ok(k in cert, k + ' missing');
  }
  for (const k of ['strategies', 'comparator', 'uncertainty', 'intersectional']) {
    assert.ok(k in cert.results, 'results.' + k + ' missing');
  }
});

test('buildCertificate digest is the sha256 of its canonical payload', () => {
  const cert = buildCertificate(auditFixture());
  const payload = { ...cert };
  delete payload.sha256;
  assert.equal(cert.sha256, sha256Hex(canonicalize(payload)));
  assert.match(cert.sha256, /^[0-9a-f]{64}$/);
});

test('buildCertificate is deterministic for identical audit inputs', () => {
  const a = buildCertificate(auditFixture());
  const b = buildCertificate(auditFixture());
  assert.equal(a.sha256, b.sha256);
  assert.equal(certificateJson(a), certificateJson(b));
});

test('buildCertificate digest changes with inputs, not with wall time', () => {
  const base = buildCertificate(auditFixture()).sha256;
  for (const opts of [{ seed: 4243 }, { bias: 0.9 }, { threshold: 0.6 }, { scenarioId: 'shelter' }]) {
    const other = buildCertificate(auditFixture(opts));
    assert.notEqual(other.sha256, base, JSON.stringify(opts));
  }
  // Same inputs audited again later: identical digest, no timestamps.
  assert.equal(buildCertificate(auditFixture()).sha256, base);
});

test('buildCertificate carries the honesty disclaimer verbatim', () => {
  const cert = buildCertificate(auditFixture());
  assert.equal(cert.disclaimer, DATA_STATEMENT);
  assert.equal(cert.syntheticDataOnly, true);
  assert.ok(cert.disclaimer.includes('synthetic'));
  assert.ok(cert.disclaimer.includes('No patient, student, or shelter guest records'));
  assert.ok(/teaching tool/.test(cert.disclaimer));
});

test('buildCertificate inputs echo the exact scenario parameters', () => {
  const cert = buildCertificate(auditFixture({ seed: 99, bias: 0.25, threshold: 0.3 }));
  assert.equal(cert.input.seed, 99);
  assert.equal(cert.input.bias, 0.25);
  assert.equal(cert.input.threshold, 0.3);
  assert.equal(cert.input.scenario, 'clinic');
});

test('buildCertificate comparator section mirrors the workbench policies', () => {
  const audit = auditFixture();
  const cert = buildCertificate(audit);
  assert.equal(cert.results.comparator.length, audit.workbench.comparator.policies.length);
  for (const p of cert.results.comparator) {
    const wb = audit.workbench.comparator.policies.find((w) => w.id === p.id);
    assert.ok(wb, p.id);
    assert.equal(p.recall, wb.eval.recall);
    assert.equal(p.missed, wb.eval.counts.fn);
    assert.equal(p.alerts, wb.eval.counts.tp + wb.eval.counts.fp);
    assert.equal(p.eoGap, wb.eval.gaps.equalOpportunity);
  }
});

test('buildCertificate intersectional section mirrors slice cells', () => {
  const audit = auditFixture();
  const cert = buildCertificate(audit);
  assert.equal(cert.results.intersectional.cells.length, audit.workbench.slices.cells.length);
  assert.equal(cert.results.intersectional.minN, audit.workbench.slices.minN);
  const flagged = cert.results.intersectional.cells.filter((c) => c.smallCell).length;
  assert.equal(flagged, audit.workbench.slices.smallCellCount);
});

test('buildCertificate evidence digest moves when predictions move', () => {
  const a = auditFixture();
  const cert1 = buildCertificate(a);
  const mutated = JSON.parse(JSON.stringify(a));
  const s = mutated.strategies[0];
  s.predictions[0] = s.predictions[0] ? 0 : 1;
  const cert2 = buildCertificate(mutated);
  assert.notEqual(cert1.evidence.scoreDigest, cert2.evidence.scoreDigest);
  assert.notEqual(cert1.sha256, cert2.sha256);
});

/* ------------------------------------------------------------------ */
/* Certificate serializations                                          */
/* ------------------------------------------------------------------ */

test('certificateJson round-trips and embeds the digest', () => {
  const cert = buildCertificate(auditFixture());
  const parsed = JSON.parse(certificateJson(cert));
  assert.equal(parsed.sha256, cert.sha256);
  assert.equal(parsed.format, CERT_FORMAT);
});

test('certificateCsv stays injection safe and carries the digest in a comment', () => {
  const cert = buildCertificate(auditFixture());
  const csv = certificateCsv(cert);
  assert.ok(csv.startsWith('# ' + DATA_STATEMENT));
  assert.ok(csv.includes('# integrity sha256: ' + cert.sha256));
  const body = csv.split(/\r?\n/).filter((l) => !l.startsWith('#')).join('\n');
  const parsed = parseCsv(body);
  assert.ok(parsed.rows.length >= 3, 'needs the strategy rows');
  for (const row of parsed.rows) {
    for (const v of Object.values(row)) {
      const s = String(v);
      assert.ok(!/^[=+\-@\t\r]/.test(s) || s.startsWith("'"), 'unsafe cell ' + s);
    }
  }
});

test('certificateCsv reports every audited strategy', () => {
  const cert = buildCertificate(auditFixture());
  const csv = certificateCsv(cert);
  for (const s of cert.results.strategies) {
    assert.ok(csv.includes(s.strategy), s.strategy + ' absent from CSV');
  }
});

test('certificateMarkdown embeds digest, disclaimer and honesty limits', () => {
  const cert = buildCertificate(auditFixture());
  const md = certificateMarkdown(cert);
  assert.ok(md.includes(cert.sha256));
  assert.ok(md.includes(DATA_STATEMENT));
  assert.ok(/not a\s+cryptographic signature/i.test(md), 'must not overclaim signing');
  assert.ok(/no private key exists/i.test(md));
  for (const p of cert.results.comparator) {
    assert.ok(md.includes(p.id), p.id);
  }
});

test('certificateMarkdown flags small cells instead of hiding them', () => {
  const cert = buildCertificate(runAudit({ scenarioId: 'clinic', seed: 3, n: 60 }));
  const md = certificateMarkdown(cert);
  assert.ok(/Small-cell caution/.test(md), 'markdown must surface small-cell warnings');
  assert.ok(/n=30/.test(md), 'markdown must state the reliability floor');
});

/* ------------------------------------------------------------------ */
/* URL-hash bookmarking                                                */
/* ------------------------------------------------------------------ */

test('encodeAuditParams produces a deterministic audit= bookmark', () => {
  const enc = encodeAuditParams({ scenario: 'clinic', bias: 0.6, threshold: 0.35, seed: 20270101 });
  assert.ok(enc.startsWith('audit='));
  assert.match(enc.slice(6), /^[A-Za-z0-9_-]+$/);
  assert.equal(enc, encodeAuditParams({ scenario: 'clinic', bias: 0.6, threshold: 0.35, seed: 20270101 }));
});

test('bookmark round-trips for every scenario and edge parameter values', () => {
  const cases = [
    { scenario: 'clinic', bias: 0.6, threshold: 0.35, seed: 20270101 },
    { scenario: 'scholarship', bias: 0, threshold: 0.1, seed: 1 },
    { scenario: 'shelter', bias: 0.9, threshold: 0.8, seed: 999999999 }
  ];
  for (const c of cases) {
    const dec = decodeAuditParams(encodeAuditParams(c));
    assert.equal(dec.scenario, c.scenario);
    assert.equal(dec.bias, c.bias);
    assert.equal(dec.threshold, c.threshold);
    assert.equal(dec.seed, c.seed);
  }
});

test('decodeAuditParams accepts hash, audit= and bare payload forms', () => {
  const enc = encodeAuditParams({ scenario: 'clinic', bias: 0.5, threshold: 0.5, seed: 7 });
  const bare = enc.slice(6);
  for (const form of [enc, '#' + enc, bare]) {
    const dec = decodeAuditParams(form);
    assert.equal(dec.seed, 7, form.slice(0, 20));
  }
});

test('decodeAuditParams rejects every malformed or out-of-range input', () => {
  const bads = [
    '', '#', 'audit=', 'audit=!!!', 'not-a-bookmark', '#other=abc',
    encodeAuditParams({ scenario: 'clinic', bias: 0.5, threshold: 0.5, seed: 7 }) + 'x'.repeat(5000)
  ];
  // Hand-crafted invalid payloads.
  const mk = (o) => 'audit=' + base64urlEncode(JSON.stringify(o));
  bads.push(mk({ s: 'mars', b: 0.5, t: 0.5, seed: 7 }));
  bads.push(mk({ s: 'clinic', b: -0.1, t: 0.5, seed: 7 }));
  bads.push(mk({ s: 'clinic', b: 1.5, t: 0.5, seed: 7 }));
  // Outside slider range even though inside the engine's 0..1 bounds:
  // the control could not represent it, so the link must not decode.
  bads.push(mk({ s: 'clinic', b: 0.95, t: 0.5, seed: 7 }));
  bads.push(mk({ s: 'clinic', b: 0.5, t: -1, seed: 7 }));
  bads.push(mk({ s: 'clinic', b: 0.5, t: 0.05, seed: 7 }));
  bads.push(mk({ s: 'clinic', b: 0.5, t: 0.85, seed: 7 }));
  bads.push(mk({ s: 'clinic', b: 0.5, t: 0.5, seed: 0 }));
  bads.push(mk({ s: 'clinic', b: 0.5, t: 0.5, seed: 3.5 }));
  bads.push(mk({ s: 'clinic', b: 0.5, t: 0.5, seed: 2e9 }));
  bads.push(mk({ s: 'clinic', b: 0.5, t: 0.5, seed: 'seven' }));
  bads.push(mk({ s: 'clinic' }));
  bads.push(mk([1, 2, 3]));
  bads.push(mk('a string'));
  bads.push('audit=' + base64urlEncode('{broken json'));
  for (const b of bads) {
    assert.equal(decodeAuditParams(b), null, JSON.stringify(b).slice(0, 60));
  }
});

test('decodeAuditParams tolerates unknown extra fields for forward compat', () => {
  const mk = (o) => 'audit=' + base64urlEncode(JSON.stringify(o));
  const dec = decodeAuditParams(mk({ s: 'clinic', b: 0.5, t: 0.5, seed: 7, future: 'field' }));
  assert.ok(dec);
  assert.equal(dec.seed, 7);
});

test('decoded bookmark reproduces the identical audit digest', () => {
  const params = { scenario: 'shelter', bias: 0.7, threshold: 0.55, seed: 31337 };
  const dec = decodeAuditParams(encodeAuditParams(params));
  const a = buildCertificate(runAudit({ scenarioId: params.scenario, bias: params.bias, threshold: params.threshold, seed: params.seed, n: 400 }));
  const b = buildCertificate(runAudit({ scenarioId: dec.scenario, bias: dec.bias, threshold: dec.threshold, seed: dec.seed, n: 400 }));
  assert.equal(a.sha256, b.sha256);
});

/* ------------------------------------------------------------------ */
/* runSelfTests                                                        */
/* ------------------------------------------------------------------ */

test('runSelfTests returns a non-empty list of named, structured checks', () => {
  const checks = runSelfTests();
  assert.ok(checks.length >= 15, 'self-test count ' + checks.length);
  const ids = new Set();
  for (const c of checks) {
    assert.equal(typeof c.id, 'string');
    assert.equal(typeof c.name, 'string');
    assert.equal(typeof c.pass, 'boolean');
    assert.equal(typeof c.detail, 'string');
    assert.ok(!ids.has(c.id), 'duplicate id ' + c.id);
    ids.add(c.id);
  }
});

test('runSelfTests covers the six mandated invariant classes', () => {
  const checks = runSelfTests();
  const ids = checks.map((c) => c.id).join(' ');
  for (const probe of ['ece', 'brier', 'wilson', 'monoton', 'small', 'sha256']) {
    const found = checks.some((c) => c.id.includes(probe) || c.name.toLowerCase().includes(probe));
    assert.ok(found, 'no self-test mentions ' + probe + ' in ' + ids);
  }
});

test('runSelfTests passes cleanly with and without an audit object', () => {
  const a = runAudit({ seed: 11, n: 300 });
  const withAudit = runSelfTests(a);
  const without = runSelfTests();
  for (const c of withAudit) assert.ok(c.pass, c.id + ' failed: ' + c.detail);
  for (const c of without) assert.ok(c.pass, c.id + ' failed: ' + c.detail);
});

test('runSelfTests is deterministic in pass state and order', () => {
  const a = runSelfTests().map((c) => c.id + ':' + c.pass).join(',');
  const b = runSelfTests().map((c) => c.id + ':' + c.pass).join(',');
  assert.equal(a, b);
});

/* ------------------------------------------------------------------ */
/* UI source invariants for the new workbench                          */
/* ------------------------------------------------------------------ */

test('index.html mounts every new workbench and certificate element', () => {
  const html = read('index.html');
  for (const id of [
    'reliability-svg', 'frontier-svg', 'overlay-groups', 'frontier-legend',
    'comparator-table', 'slices-table',
    'cert-hash', 'cert-json', 'cert-csv', 'cert-md', 'copy-link',
    'run-selftests', 'selftest-list'
  ]) {
    assert.ok(html.includes('id="' + id + '"'), id + ' missing from index.html');
  }
});

test('index.html keeps at most two aria-live regions after the new sections', () => {
  const html = read('index.html');
  const liveCount = (html.match(/aria-live=/g) || []).length;
  assert.ok(liveCount <= 2, 'aria-live count ' + liveCount);
});

test('new buttons are non-submit buttons inside the page', () => {
  const html = read('index.html');
  for (const id of ['cert-json', 'cert-csv', 'cert-md', 'copy-link', 'run-selftests']) {
    const re = new RegExp(`<button[^>]*id="${id}"[^>]*>`);
    const m = html.match(re);
    assert.ok(m, id + ' button missing');
    assert.ok(m[0].includes('type="button"'), id + ' lacks type="button"');
  }
});

test('app.js wires every new control and the hashchange lifecycle', () => {
  const src = read('src/app.js');
  for (const token of [
    'run-selftests', 'cert-json', 'cert-csv', 'cert-md', 'copy-link',
    'overlay-groups', 'hashchange', 'decodeAuditParams', 'encodeAuditParams',
    'buildCertificate', 'runSelfTests', 'drawReliabilitySvg', 'drawFrontierSvg'
  ]) {
    assert.ok(src.includes(token), token + ' not referenced in app.js');
  }
});

/* Strips comments so scans below check real code, not the words inside
 * a comment like "no innerHTML". */
function codeOnly(src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|\s)\/\/[^\n]*/g, ' ');
}

test('new modules never use innerHTML, document.write or eval', () => {
  for (const f of ['src/calibration.js', 'src/audit-cert.js', 'src/svg-charts.js', 'src/app.js']) {
    const src = codeOnly(read(f));
    for (const bad of ['innerHTML', 'outerHTML', 'document.write', 'eval(', 'insertAdjacentHTML']) {
      assert.ok(!src.includes(bad), f + ' uses ' + bad);
    }
  }
});

test('svg-charts builds nodes through createElementNS only', () => {
  const src = codeOnly(read('src/svg-charts.js'));
  assert.ok(src.includes('createElementNS'));
  assert.ok(!src.includes('innerHTML'));
});

test('no em or en dashes slipped into the new files', () => {
  for (const f of ['src/calibration.js', 'src/audit-cert.js', 'src/svg-charts.js', 'tests/wave6-workbench.test.mjs', 'tests/wave7-cert.test.mjs']) {
    const src = read(f);
    assert.ok(!src.includes('\u2014') && !src.includes('\u2013'), f + ' contains a dash');
  }
  const html = read('index.html');
  assert.ok(!html.includes('\u2014') && !html.includes('\u2013'), 'index.html dash');
});

test('the strict CSP is unchanged and forbids external connections', () => {
  const html = read('index.html');
  assert.ok(html.includes("default-src 'none'"), 'CSP default-src weakened');
  assert.ok(html.includes("connect-src 'none'"), 'CSP connect-src weakened');
  assert.ok(html.includes("script-src 'self'"), 'CSP script-src weakened');
});

test('no network endpoints or CDN URLs were introduced in new files', () => {
  for (const f of ['src/calibration.js', 'src/audit-cert.js', 'src/svg-charts.js']) {
    const src = codeOnly(read(f));
    for (const pat of ['https://', 'http://', 'fetch(', 'XMLHttpRequest', 'WebSocket', 'sendBeacon']) {
      // The SVG XML namespace identifier is a URI constant, not a request.
      if ((pat === 'http://' || pat === 'https://') && f === 'src/svg-charts.js') {
        const withoutNs = src.replace(/['"]http:\/\/www\.w3\.org\/2000\/svg['"]/g, "''");
        assert.ok(!withoutNs.includes(pat), f + ' contains ' + pat);
        continue;
      }
      assert.ok(!src.includes(pat), f + ' contains ' + pat);
    }
  }
});

test('app.js still leaves honesty metadata and disclaimers intact', () => {
  const html = read('index.html');
  assert.ok(html.includes('synthetic'), 'synthetic disclosure missing');
  const engine = read('src/ml-engine.js');
  assert.ok(engine.includes('DATA_STATEMENT'));
  const cert = read('src/audit-cert.js');
  assert.ok(cert.includes('DATA_STATEMENT'), 'certificate must carry the disclosure');
});

test('audit object remains fully JSON-serializable with the workbench', () => {
  const a = runAudit({ seed: 2, n: 300 });
  const text = JSON.stringify(a);
  const parsed = JSON.parse(text);
  assert.equal(parsed.workbench.uncertainty.n, a.workbench.uncertainty.n);
  assert.ok(text.length > 1000);
});
