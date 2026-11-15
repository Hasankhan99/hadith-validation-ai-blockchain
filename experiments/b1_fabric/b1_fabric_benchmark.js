/* B1: functional and performance evaluation of the v2 chaincode on the Hyperledger Fabric 2.5.16
 * test network (2 orgs, 1 Raft orderer, default batch timeout, default majority endorsement policy).
 * Payloads are synthetic (random verdicts, hashed placeholder strings); model digests of LR and RF are the
 * SHA-256 of the saved v1 .joblib files (when MODELS_DIR is set), the remaining ones are digests of labelled
 * placeholder strings.
 * Usage: node b1_fabric_benchmark.js <fabric-samples/test-network dir> <out dir> [N_SEQ] [N_CONC] */
const grpc = require('@grpc/grpc-js');
const { connect, signers } = require('@hyperledger/fabric-gateway');
const crypto = require('crypto'), fs = require('fs'), path = require('path');

const NET = process.argv[2], OUT = process.argv[3];
const N_SEQ = +(process.argv[4] || 200), N_CONC = +(process.argv[5] || 300);
const MODELS_DIR = process.env.MODELS_DIR;
const dec = new TextDecoder();
const sha = s => crypto.createHash('sha256').update(s).digest('hex');
const fileSha = f => crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');

function open(org, port) {
  const base = path.join(NET, 'organizations/peerOrganizations', `org${org}.example.com`);
  const first = d => path.join(d, fs.readdirSync(d)[0]);
  const cert = fs.readFileSync(first(path.join(base, `users/User1@org${org}.example.com/msp/signcerts`)));
  const key = crypto.createPrivateKey(fs.readFileSync(first(path.join(base, `users/User1@org${org}.example.com/msp/keystore`))));
  const tls = fs.readFileSync(path.join(base, `peers/peer0.org${org}.example.com/tls/ca.crt`));
  const client = new grpc.Client(`localhost:${port}`, grpc.credentials.createSsl(tls),
    { 'grpc.ssl_target_name_override': `peer0.org${org}.example.com` });
  const gw = connect({ client, identity: { mspId: `Org${org}MSP`, credentials: cert }, signer: signers.newPrivateKeySigner(key),
    evaluateOptions: () => ({ deadline: Date.now() + 10000 }), endorseOptions: () => ({ deadline: Date.now() + 30000 }),
    submitOptions: () => ({ deadline: Date.now() + 30000 }), commitStatusOptions: () => ({ deadline: Date.now() + 60000 }) });
  return { gw, client, contract: gw.getNetwork('hadithchannel').getContract('hadith') };
}

const M = ['LR', 'RF', 'SVM', 'CAMeLBERT', 'AraBERT'];
const digests = {
  LR: MODELS_DIR ? fileSha(path.join(MODELS_DIR, 'model_logistic_regression.joblib')) : sha('placeholder-LR'),
  RF: MODELS_DIR ? fileSha(path.join(MODELS_DIR, 'model_random_forest.joblib')) : sha('placeholder-RF'),
  SVM: sha('placeholder-SVM'), CAMeLBERT: sha('placeholder-CAMeLBERT'), AraBERT: sha('placeholder-AraBERT') };
const run = Date.now().toString(36);
const verdictsFor = () => Object.fromEntries(M.map(m => [m, crypto.randomInt(0, 3)]));
const args = (id, over = {}) => {
  const v = over.verdicts || verdictsFor(), h = over.hashes || { ...digests };
  return [over.txid || `${run}-${id}`, `H-${id}`, sha(`hadith-${id}`), String(over.res ?? 2), String(over.conf ?? 0.91),
          over.status || 'VERIFIED', JSON.stringify(v), JSON.stringify(h)];
};
async function expectFail(name, fn, needle, results) {
  let msg = null;
  try { await fn(); } catch (e) { msg = (e.details && e.details.map(d => d.message).join(' | ')) || e.message; }
  const ok = msg !== null && (!needle || msg.includes(needle));
  results.push({ test: name, rejected: msg !== null, expected_message_found: ok, message: msg });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  -> ${msg}`);
}
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const stats = a => ({ n: a.length, mean_ms: a.reduce((x, y) => x + y, 0) / a.length, p50_ms: pct(a, .5), p95_ms: pct(a, .95), p99_ms: pct(a, .99), max_ms: Math.max(...a) });

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const o1 = open(1, 7051), o2 = open(2, 9051), R = { run, digests, security: [], note: 'synthetic payloads; see header' };
  for (const m of M) await o1.contract.submitTransaction('RegisterModel', m, m, digests[m], '0.95');
  console.log('registered', M.length, 'models');

  // functional / security tests
  await o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args('ok1'));
  const rec = JSON.parse(dec.decode(await o1.contract.evaluateTransaction('GetValidation', `${run}-ok1`)));
  R.security.push({ test: 'valid consensus record committed and retrievable', rejected: false, expected_message_found: rec.consensus_label === 'Sahih', message: null });
  console.log('PASS  valid record committed', rec.consensus_label, rec.fabric_tx_id.slice(0, 12));
  const flip = { ...digests, LR: (digests.LR[0] === 'a' ? 'b' : 'a') + digests.LR.slice(1) };
  await expectFail('one hex digit of the LR digest altered', () => o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args('t1', { hashes: flip })), 'tampering detected', R.security);
  const omit = { ...digests }; delete omit.RF;
  await expectFail('registered model (RF) omitted from submission', () => o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args('t2', { hashes: omit })), 'missing from submission', R.security);
  await expectFail('unregistered model added to submission', () => o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args('t3', { hashes: { ...digests, EVIL: sha('evil') }, verdicts: { ...verdictsFor(), EVIL: 0 } })), 'not registered', R.security);
  await expectFail('verdict missing for a registered model', () => { const v = verdictsFor(); delete v.SVM; return o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args('t4', { verdicts: v })); }, 'missing verdict', R.security);
  await expectFail('replay of an existing validation id', () => o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args('ok1')), 'already recorded', R.security);
  await expectFail('consensus label outside {0,1,2}', () => o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args('t5', { res: 3 })), 'invalid consensus result', R.security);
  await expectFail('confidence outside [0,1]', () => o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args('t6', { conf: 1.5 })), 'outside [0,1]', R.security);
  await expectFail('model registration by non-admin organisation (Org2)', () => o2.contract.submitTransaction('RegisterModel', 'ROGUE', 'ROGUE', sha('rogue'), '0.9'), 'access denied', R.security);
  R.security_pass = R.security.filter(r => r.expected_message_found).length; R.security_total = R.security.length;

  // sequential submit latency (submit = endorse + order + commit wait)
  const seq = [];
  for (let i = 0; i < N_SEQ; i++) { const t = performance.now(); await o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args(`s${i}`)); seq.push(performance.now() - t); }
  R.sequential_submit = stats(seq);
  fs.writeFileSync(path.join(OUT, 'b1_seq_latency_ms.csv'), 'i,ms\n' + seq.map((x, i) => `${i},${x.toFixed(2)}`).join('\n'));
  console.log('sequential', R.sequential_submit);

  // concurrent throughput: N_CONC transactions, C in flight
  R.concurrent = [];
  for (const C of [10, 50]) {
    const lat = []; let next = 0, failed = 0; const t0 = performance.now();
    await Promise.all(Array.from({ length: C }, async () => {
      while (true) { const i = next++; if (i >= N_CONC) return; const t = performance.now();
        try { await o1.contract.submitTransaction('VerifyAndRecordConsensus', ...args(`c${C}-${i}`)); lat.push(performance.now() - t); } catch (e) { failed++; } }
    }));
    const el = (performance.now() - t0) / 1000;
    R.concurrent.push({ concurrency: C, submitted: N_CONC, committed: lat.length, failed, elapsed_s: el, tps: lat.length / el, latency: stats(lat) });
    console.log('concurrency', C, R.concurrent.at(-1).tps.toFixed(2), 'tps', 'failed', failed);
  }

  // query latency
  const q = [];
  for (let i = 0; i < 300; i++) { const t = performance.now(); await o1.contract.evaluateTransaction('GetValidation', `${run}-s${i % N_SEQ}`); q.push(performance.now() - t); }
  R.query = stats(q);
  fs.writeFileSync(path.join(OUT, 'b1_fabric_results.json'), JSON.stringify(R, null, 2));
  console.log('query', R.query); o1.gw.close(); o2.gw.close(); o1.client.close(); o2.client.close();
})().catch(e => { console.error('FATAL', e); process.exit(1); });
