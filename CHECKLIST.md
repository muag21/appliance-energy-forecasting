# Pre-submission checklist

Three layers of verification. Only the first is automated.

| Layer | Method | Catches |
|---|---|---|
| 1. Mechanical | `python scripts/verify.py` | Missing files, failing tests, placeholders, inconsistent numbers |
| 2. Reproducibility | Clean-clone run | "Works on my machine" failures |
| 3. Judgement | This checklist | The things that actually carry marks |

---

## Layer 1 — Automated

```bash
python scripts/verify.py
```

Twenty-four checks. Exit status 0 only when all pass. Run it last, after every
edit, not once at the start.

---

## Layer 2 — Reproducibility from clean

The README claims the repository runs from a fresh clone. Verify that claim
rather than assuming it. In a directory that is **not** your working copy:

```bash
git clone <your-repo-url> verify-clone
cd verify-clone

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                  # must pass with no prior setup
python scripts/run_pipeline.py          # must download data and complete
python scripts/verify.py
```

Things that commonly break here and pass in your working copy:

- [ ] Data downloads without manual intervention (or the README says clearly what to do if it doesn't)
- [ ] No absolute paths from your machine anywhere in the code
- [ ] `requirements.txt` actually lists everything imported — a package you installed months ago for something else will not be there
- [ ] Notebooks run top to bottom in a fresh kernel, in order
- [ ] Figures regenerate rather than being loaded from committed PNGs
- [ ] Random seeds produce identical metrics on a second run

If the second run gives different numbers, find out why before submitting. An
unseeded component is a reproducibility failure regardless of how good the
results are.

---

## Layer 3 — Judgement

### Against the brief

Work through the assignment specification line by line and mark where each
requirement is met. Not "I think I covered that" — a section number.

- [ ] Target, horizon, test period as specified
- [ ] All five benchmarks
- [ ] SARIMAX with exogenous covariates
- [ ] Feature-based ML with lags, rolling stats, cyclic time, sensors
- [ ] Foundation model
- [ ] MAE, RMSE, MASE, Bias for every model on the same test period
- [ ] All six mandatory questions answered explicitly, not implied
- [ ] Report within the page limit
- [ ] Every required repository artefact present

### Defensibility

For each claim in the report, ask: **could I defend this if challenged?**

- [ ] Every number in the report traces to a file in `outputs/`
- [ ] Every figure is referenced in the text and adds something the text doesn't
- [ ] No claim of statistical significance without a test behind it
- [ ] No causal language where only correlation was established
- [ ] Limitations section names real weaknesses, not token ones
- [ ] Conditional and operational forecasts distinguished everywhere, not just once

### The viva test

Three questions an examiner is likely to ask about this specific report. If you
cannot answer them at a whiteboard without notes, that part isn't yours yet.

- [ ] **Why direct multi-horizon rather than recursive prediction?** (Error accumulation; train/apply distribution shift on lag inputs)
- [ ] **You report a 13.7% improvement over the benchmark and then recommend the benchmark. Justify that.** (*p* = 0.32; one fortnight; serially correlated losses)
- [ ] **What was the validation-split leak, and why didn't origin-anchoring prevent it?** (Random split across 24 near-duplicate rows per timestamp; feature-matrix guarantees don't extend to library row partitioning)

Add your own: for each section, write the one question you'd least like to be
asked, then answer it.

### Authorship

- [ ] Every paragraph is in your words, not reworded from a draft you were given
- [ ] You can explain why each analytical choice was made, not just what it was
- [ ] Points where you disagree with a suggested framing are argued your way
- [ ] Institutional AI-use declaration completed honestly
- [ ] Every reference checked against the actual source — authors, year, journal, pages

Reading a paragraph, closing it, and rewriting from memory is the test. If you
can't reconstruct the argument, you don't yet own it.

---

## Known outstanding items

As of the last verification run:

1. **Section 8 (foundation model)** — not run. Requires `pip install torch chronos-forecasting` and network access to the model host.
2. **Mandatory question 4** — depends on item 1.
3. **Draft placeholders** — 4 remaining, including the title block.
4. **Report length** — ~7,600 words against a 6–8 page target. See README notes on what to cut.
5. **Grid search** — `--tune` never run to completion; the iteration count is selected chronologically, other hyperparameters are at defaults.
6. **Expanding-window backtest** — the single largest methodological gap, acknowledged in Section 10 but not addressed.
