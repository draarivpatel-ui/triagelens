# TriageLens

**Live auditor:** [https://aariv.me/triagelens/](https://aariv.me/triagelens/)

**An interactive, zero-dependency, 100% browser-local ML classifier, calibration and algorithmic fairness auditor.**

Built for the Teens in AI Global Challenge 2027 (Ages 13+, entries close 15 February 2027). TriageLens trains two real classifiers in the browser, calibrates their probabilities, and shows the exact mathematical trade-off between overall recall and subgroup fairness. All data is **synthetic** and generated inside the page. There are no real patient, student, or shelter records, no pilots, no analytics, no CDN, and no network requests.

## Run it

Serve the folder over local HTTP, then open the URL in any modern browser. ES modules do **not** load over `file://`, so opening `index.html` directly will not work.

```bash
# From the folder containing this README:
python3 -m http.server 8000
# then open http://localhost:8000/
# If you serve the parent folder instead, open http://localhost:8000/triagelens/
```

There is no build step and no install. Run the deterministic unit tests with:

```bash
node --test tests/ml-engine.test.mjs
```

## What it does

1. **Three synthetic benchmark scenarios**
   - Community Clinic Triage Priority: under-triage of rural / low-access patients.
   - Student Scholarship and Tutoring Support: zip-code proxy bias.
   - Emergency Shelter Bed Routing: language-barrier reporting bias.
2. **Two real models trained from scratch** on a stratified 70/30 train/held-out split (1,400 seeded rows by default):
   - L2-regularized logistic regression with full-batch gradient descent, standardization and a sampled real loss curve.
   - Depth-limited Gini decision tree with interpretable rules.
3. **Calibration** - Platt / temperature scaling fitted on **training rows only**; reported on held-out rows. A guard keeps the identity mapping if a fit would worsen the combined Brier + ECE objective. Proxy removal fits its own train-only calibration.
4. **Four mitigation policies plus an uncalibrated baseline on the same held-out rows**: calibrated global threshold, equal-opportunity thresholds, selection-rate thresholds, and proxy-feature removal.
5. **Exact held-out metrics**: accuracy, precision, recall, F1, ROC-AUC, Brier score, 10-bin ECE, plus per-subgroup selection rate (demographic parity gap), TPR (equal opportunity gap), FNR disparity, and subgroup calibration error.
6. **Explicit trade-off reporting**: a live banner names every mitigation that lowers overall recall or widens a secondary gap, so no cost is hidden.
7. **Counterexample inspector**: specific held-out positives from the harmed subgroup that the global threshold misses but a mitigation recovers.
8. **Model card and ethics panel**: who benefits, who could be harmed, likely failure modes, and why a human must decide.

## Observed benchmark output

`node --test tests/ml-engine.test.mjs` (Node v26.8.1):

```text
ℹ tests 54
ℹ suites 0
ℹ pass 54
ℹ fail 0
ℹ cancelled 0
ℹ skipped 0
ℹ todo 0
ℹ duration_ms ~3000 (indicative)
```

`runAudit()` at seed 20270101, bias 0.60, threshold 0.35, n = 1400 (all numbers below are generated directly from live code output; timings are indicative only and vary by machine):

```text
scenario     engine ms  LR acc  AUC     Brier   ECE     tree acc  majority
clinic           ~54     0.7190  0.7722  0.1628  0.0395   0.7310    0.7286
scholarship      ~37     0.7643  0.8221  0.1420  0.0471   0.7595    0.7381
shelter          ~54     0.7000  0.7928  0.1857  0.0552   0.6833    0.5595

calibration (train split): clinic ECE 0.0357 -> 0.0302, Brier 0.1555 -> 0.1554
                           scholarship ECE 0.0395 -> 0.0300, Brier 0.1480 -> 0.1480
                           shelter: identity kept (fit did not improve combined objective)

equal opportunity gap (held out):
clinic       0.2465 -> 0.2238   (FNR 0.2465 -> 0.2238),   3 counterexamples
scholarship  0.3765 -> 0.0607   (FNR 0.3765 -> 0.0607),   6 counterexamples
shelter      0.2867 -> 0.0514   (FNR 0.2867 -> 0.0514),  12 counterexamples

overall recall, global threshold -> equal opportunity:
clinic       0.5789 -> 0.5965   (selection-rate gap widens by 3.6 pp)
scholarship  0.6273 -> 0.6818
shelter      0.8811 -> 0.9243
```

All engine timings above are indicative measurements from one representative run, not guarantees. A full retrain and re-evaluation with all charts typically completes in tens of milliseconds. The interface debounces slider input by 90 ms so dragging stays smooth.

## Fairness method

The equal-opportunity mitigation implements post-processing after Hardt, Price and Srebro (2016), with a deliberate anti-leveling-down change. Every threshold is fitted on **calibrated training scores only**, in the exact same probability space used for held-out evaluation. The target is the higher TPR observed at the group's own accuracy-maximizing operating point or at the global working threshold; every group's threshold is then chosen to reach that target TPR instead of being dragged down to the lowest group's accuracy-maximizing TPR. This keeps overall recall at or above the global baseline: at the documented defaults, equal opportunity raises recall by 1.8 pp (clinic), 5.5 pp (scholarship), and 4.3 pp (shelter).

Selection-rate mitigation picks per-group thresholds nearest a selection target computed strictly from calibrated **train** rows. Proxy removal retrains without the named proxy feature and fits its own train-only calibration. No threshold, target, or calibration parameter reads a held-out row. The live trade-off banner reports any mitigation that changes overall recall or widens a secondary gap (for example, clinic equal opportunity lifts recall but widens the selection-rate gap by 3.6 pp).

## Ethics and limits

- Every record is synthetic, seeded, and generated in the browser. No real-world deployment or pilot exists.
- Fairness gaps compare only the two groups in each benchmark; real fairness audits need far more context.
- Equal opportunity can change overall accuracy and secondary gaps. Every such change is shown in the trade-off banner, not hidden.
- Removing one proxy does not erase bias carried by correlated features.
- Small subgroups make observed error rates noisy. At the clinic default, train and held-out TPR curves differ enough that a train-only equal-opportunity fit can only close a small part of the observed gap; the banner reports the residual honestly.
- **A human must decide.** TriageLens is a teaching tool, never a clinical, educational, or shelter-placement decision system.

## Files

```text
index.html                 markup and accessible structure
styles.css                 editorial light/dark theme, responsive layout
src/ml-engine.js           PRNG, generators, models, calibration, fairness, metrics
src/charts.js              Canvas ROC, reliability, disparity, loss charts
src/app.js                 UI controller, exports, accessibility wiring
tests/ml-engine.test.mjs   54 deterministic Node tests
video/narration.txt        2-5 minute demo narration
video/steps.json           verified demo click/tap choreography
```

## License

MIT. See `LICENSE`.
