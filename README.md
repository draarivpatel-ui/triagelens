# TriageLens

**An interactive, zero-dependency, 100% browser-local ML classifier, calibration and algorithmic fairness auditor.**

Built for the Teens in AI Global Challenge 2027 (Ages 13+, entries close 15 February 2027). TriageLens trains two real classifiers in the browser, calibrates their probabilities, and shows the exact mathematical trade-off between overall accuracy and subgroup fairness. All data is **synthetic** and generated inside the page. There are no real patient, student, or shelter records, no pilots, no analytics, no CDN, and no network requests.

## Run it

Open `index.html` in any modern browser. There is no build step and no install.

```bash
# Optional local server (still fully offline):
python3 -m http.server 8000
# then open http://localhost:8000
```

Run the deterministic unit tests:

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
3. **Calibration** - Platt / temperature scaling fitted on **training rows only**; reported on held-out rows. A guard keeps the identity mapping if a fit would worsen the combined Brier + ECE objective.
4. **Four decision policies on the same held-out rows**: uncalibrated baseline, calibrated global threshold, equal-opportunity thresholds, selection-rate thresholds, and proxy-feature removal.
5. **Exact held-out metrics**: accuracy, precision, recall, F1, ROC-AUC, Brier score, 10-bin ECE, plus per-subgroup selection rate (demographic parity gap), TPR (equal opportunity gap), FNR disparity, and subgroup calibration error.
6. **Counterexample inspector**: specific held-out positives from the harmed subgroup that the global threshold misses but a mitigation recovers.
7. **Model card and ethics panel**: who benefits, who could be harmed, likely failure modes, and why a human must decide.

## Observed benchmark output

`node --test tests/ml-engine.test.mjs` (Node v26.8.1):

```text
ℹ tests 47
ℹ suites 0
ℹ pass 47
ℹ fail 0
ℹ cancelled 0
ℹ skipped 0
ℹ todo 0
ℹ duration_ms ~1050
```

`runAudit()` at seed 20270101, bias 0.60, threshold 0.35, n = 1400 (one representative run; milliseconds vary by machine):

```text
scenario     engine ms  LR acc  AUC     Brier   ECE     tree acc  majority
clinic           ~25     0.7190  0.7722  0.1628  0.0395   0.7310    0.7286
scholarship      ~23     0.7667  0.8221  0.1433  0.0511   0.7595    0.7381
shelter          ~22     0.6976  0.7928  0.1849  0.0472   0.6833    0.5595

calibration (train split): clinic ECE 0.0357 -> 0.0302, Brier 0.1555 -> 0.1554
                           scholarship ECE 0.0395 -> 0.0300, Brier 0.1480 -> 0.1480
                           shelter: identity kept (fit did not improve combined objective)

equal opportunity gap (held out):
clinic       0.2465 -> 0.0280   (FNR 0.2465 -> 0.0280),   4 counterexamples
scholarship  0.3655 -> 0.1093   (FNR 0.3655 -> 0.1093),   6 counterexamples
shelter      0.3617 -> 0.0358   (FNR 0.3617 -> 0.0358),  14 counterexamples
```

Both models are trained in roughly 15-20 ms combined on a modern laptop; a full retrain and re-evaluation with all charts typically completes in under 30 ms. The interface debounces slider input by 90 ms so dragging stays smooth.

## Fairness method

The equal-opportunity mitigation implements post-processing after Hardt, Price and Srebro (2016). For each subgroup, a threshold near its own accuracy-maximizing operating point is found on the training split. The subgroup with the lower TPR is then given the lowest threshold whose training TPR reaches the better subgroup's TPR. Selection-rate mitigation picks per-group thresholds nearest the global selection rate. Proxy removal retrains without the named proxy feature. All thresholds are fitted **without touching held-out rows**, then evaluated on them.

## Ethics and limits

- Every record is synthetic, seeded, and generated in the browser. No real-world deployment or pilot exists.
- Fairness gaps compare only the two groups in each benchmark; real fairness audits need far more context.
- Equalized opportunity can lower overall accuracy. That trade-off is shown explicitly, not hidden.
- Removing one proxy does not erase bias carried by correlated features.
- Small subgroups make observed error rates noisy.
- **A human must decide.** TriageLens is a teaching tool, never a clinical, educational, or shelter-placement decision system.

## Files

```text
index.html                 markup and accessible structure
styles.css                 editorial light/dark theme, responsive layout
src/ml-engine.js           PRNG, generators, models, calibration, fairness, metrics
src/charts.js              Canvas ROC, reliability, disparity, loss charts
src/app.js                 UI controller, exports, accessibility wiring
tests/ml-engine.test.mjs   47 deterministic Node tests
video/narration.txt        2-5 minute demo narration
video/steps.json           verified demo click/tap choreography
```

## License

MIT. See `LICENSE`.
