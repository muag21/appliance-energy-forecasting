## 8. Foundation model

### 8.1 Configuration

Chronos-T5 (small variant, approximately 46 million parameters) was applied
zero-shot. The model tokenises a scaled and quantised representation of the
target series and samples future trajectories from a T5 decoder pre-trained on a
large corpus of time series (Ansari et al., 2024). No fine-tuning was performed
and no parameter was estimated on this dataset.

A context of 512 hours preceded each origin and 20 sample paths were drawn per
forecast. The 24-step horizon sits well inside the model's native prediction
length, so no autoregressive stitching was required. The fourteen origins
completed in 53 to 60 seconds per run on Colab hardware.

**The model is univariate.** It receives only the target history — no calendar
features, no weather, no indoor sensors. Given the finding in Section 7.3 that
calendar features carry 90.7 per cent of the feature model's permutation
importance, this is a severe handicap: Chronos must infer the daily profile from
the series itself, while every competing model is handed the hour of day
directly.

### 8.2 Results

| Metric | Chronos | SARIMAX (cond.) | Feature model | Weekly seasonal naive |
|---|---|---|---|---|
| MAE | **36.23** | 37.85 | 37.89 | 43.46 |
| RMSE | 74.40 | 66.41 | **64.99** | 81.41 |
| MASE | **0.678** | 0.708 | 0.709 | 0.813 |
| Bias | −29.87 | −5.14 | −4.00 | −13.16 |

Read on MASE alone, the foundation model wins: 0.678 against 0.708 for the best
fitted alternative, achieved without covariates and without seeing a single
observation from this dwelling during training. That would be a striking result.

It is also a misleading one, for two independent reasons: the remaining columns
in the table above, and the sampling variance reported in Section 8.4.

### 8.3 The point summary drives the ranking

The bias of −29.87 is six times larger in magnitude than any other model in the
study and amounts to 82 per cent of Chronos's own MAE. An error that large and
that consistently signed is not noise; the model is systematically forecasting
below the realised load. Correspondingly, its RMSE of 74.40 is the second worst
of any non-trivial model, statistically indistinguishable from the flat mean
forecast at 74.94, and far behind SARIMAX and the feature model near 65.

The explanation lies in the point summary rather than in the model. Chronos
produces a predictive distribution; a single number must be extracted from it,
and the pointwise **median** of the sampled paths was used here. On a target
whose distribution is strongly right-skewed — mean 97.8 against median 63.3, per
Section 3.1 — the conditional median lies well below the conditional mean. The
consequence follows directly from the loss functions: absolute error is
minimised by the conditional median, squared error by the conditional mean. A
median forecast is therefore *constructed* to perform well on MAE and badly on
RMSE, and MASE, being a scaled absolute error, inherits the same bias.

The comparison is consequently not like for like. SARIMAX and the gradient
boosting model both estimate conditional means; Chronos, as configured, reports
a conditional median. Its apparent advantage on MASE partly measures that
difference in summary statistic, not a difference in forecasting ability. A fair
comparison would take the sample mean of the Chronos paths as the point
forecast, and would be expected to move the model towards the others on both
metrics — better RMSE, worse MAE. This was not run and is recommended as the
first extension in Section 10.

### 8.4 Sampling variance and interval calibration

Chronos draws sample paths stochastically, so its point forecast carries Monte
Carlo error. Repeating the full rolling-origin evaluation under three random
seeds gives:

| Seed | MASE | Runtime |
|---|---|---|
| 0 | 0.6704 | 60 s |
| 1 | 0.6971 | 54 s |
| 2 | 0.6869 | 53 s |

The spread is 0.0267 MASE; an independent repeat of the same three-seed
procedure returned 0.0273, so the variability is itself stable. **This is almost exactly the size of the gap between
Chronos and conditional SARIMAX** (0.678 against 0.708, a difference of 0.030).
Seed 1 alone would place Chronos at 0.697, effectively level with SARIMAX and
behind nothing but its own better draws.

The implication is direct: the headline figure of 0.678 reported in Section 8.2
is one realisation of a random procedure whose variability is comparable to the
effect being measured. Increasing the number of sample paths beyond 20 would
narrow the interval, and 20 was chosen for CPU tractability rather than on any
statistical grounds. Any claim that the foundation model outperforms SARIMAX
must therefore survive two separate objections — the point-summary argument in
Section 8.3, and sampling variance of the same magnitude as the claimed gap. It
survives neither.

This is worth stating in general terms. Stochastic forecasting methods should be
reported with a spread across seeds, exactly as a bootstrap estimate would be.
A single run reported as a point estimate invites precisely the over-reading
this subsection has had to correct.

**Interval calibration.** Chronos is the only model in this study producing a
predictive distribution natively, so its calibration can be assessed directly.
Empirical coverage of the nominal 80 per cent interval is **60.7 per cent**.

The intervals are therefore substantially too narrow: realised demand falls
outside them on 39 per cent of hours against an advertised 20 per cent, roughly
double the stated failure rate. For an operator sizing reserve capacity against
a stated exceedance probability, this is worse than having no interval at all,
because it supplies a specific and unwarranted level of confidence.

The likely cause is the heteroskedasticity documented in Section 9.3, where
absolute error correlates with realised load at 0.841. A model whose predictive
spread does not widen sufficiently during high-consumption periods will be
approximately calibrated overnight and badly overconfident in the evening peak,
and the aggregate figure averages the two. ⟦FILL: optional but worthwhile —
compute coverage separately for hours above and below the median load. If the
split is as described, it converts a single summary number into a diagnosis, and
it is three lines of code in notebook 06.⟧

Two observations follow. Miscalibration of this size is not repaired by drawing
more sample paths; it reflects the shape of the predictive distribution rather
than the precision with which it is estimated. And it undercuts what would
otherwise have been the strongest argument for the foundation model in this
comparison — that alone among the models considered, it quantifies its own
uncertainty without additional machinery.

### 8.5 Assessment

The honest summary is that the foundation model performs **comparably** to the
fitted models rather than better than them. Its nominal advantage on MASE
dissolves under the point-summary argument of Section 8.3 and again under the
seed variance of Section 8.4, and its one structural advantage — a native
predictive distribution — is miscalibrated by twenty percentage points. and does so without covariates or
dataset-specific training. That remains a substantive finding about zero-shot
transfer: a model that has never seen this dwelling matches one fitted to four
months of its history.

But it does not outperform them, and reporting the MASE figure without the bias
and RMSE columns would misrepresent what happened. This is a concrete instance
of the argument made throughout Section 9: a single headline metric, selected
after the fact, will support conclusions the full picture does not.

---

