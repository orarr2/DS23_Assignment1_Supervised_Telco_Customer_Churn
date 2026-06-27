# DS23 Module 3 Assignment 1 - Supervised Learning Report

Name: Orarr2

Date: 27.6.2026

Chosen task: B - Telco Customer Churn prediction (binary classification)

---

# 1. Problem framing

## 1.1 Business question

The customer retention team in the company wants to know which subscribers are
about to leave. I chose to break down the real question behind the request,
which is the calculation of the customer's value over the entire timespan in
which they are a customer of the company. This is called Customer Lifetime
Value, or LTV in short.

The churn probability of each customer translates into a calculated value
named LTV that represents a monetary measure: the customer is a function of
the money they pay over an expected period. If they pay for a subscription,
then the expected monthly payment is multiplied by the time horizon during
which they are expected to remain a customer, where the horizon is a function
of the probability of remaining a customer. The project requires building a
model that knows how to score and distinguish between all customer types, and
not just flag the obvious high-risk population.

The framing of the problem dictates many decisions later on, and most notably
the decision whether to enable the parameter `class_weight='balanced'` - so
that the probabilities remain close to true calibration, which matters to the
decision-makers of the LTV model.

## 1.2 Target variable definition

Binary label for each customer: the value is 1 if the `Churn` column equals
Yes, otherwise 0. The positive class rate stands at 26.5 percent (1,869
churners out of 7,043 customers). The dataset is a one-time cross-section,
without a time column.

## 1.3 Primary metric and business justification

I chose the F1 metric, which is a harmonic mean. The automatic choice could
have been Recall, under the assumption that FN represents the churn problem
and FP is cheap - but FP values are not cheap to the company's owners.

Every retention offer is an outreach that costs money: discount, outbound
call, customer service representative work time. The cost ratio is not
symmetric and leans towards "FN is more severe". However the ratio is not
infinite, and therefore F1 balances between the two error types.

The Accuracy value is a trap: the no-churn rate is 73.5 percent, and so a
constant model that predicts "not churning" already receives a score of 0.735
permanently, and in this case it is still useless.

ROC-AUC serves as a secondary metric, because the LTV consumer is interested
in ranking quality across the spectrum, not in a binary yes/no decision.

The decision threshold values were tuned post-training in order to maximize
F1, and the threshold landed on 0.32 for the winning model. There is a
principled pull toward a higher Recall, because F1 remains the target at that
point.

---

# 2. Results table

The test set remained locked until Part 3 and I touched it only once. The
table is sorted by F1.

| Model | F1 | ROC-AUC | Notes |
|---|---|---|---|
| baseline most_frequent | 0.000 | 0.500 | dumb model, always predicts no-churn |
| baseline stratified | 0.290 | 0.516 | dumb model, guesses by class rate |
| LR L1 at threshold 0.5 | 0.603 | 0.841 | after tuning the C parameter to 10 |
| RF at threshold 0.5 | 0.589 | 0.842 | max_depth=10, n=500, min_leaf=1 |
| XGB at threshold 0.5 | 0.590 | 0.846 | max_depth=5, lr=0.05, n=100 |
| LR L1 at tuned threshold 0.31 | 0.612 | 0.841 | threshold selected on CV |
| RF at tuned threshold 0.31 | 0.630 | 0.842 | threshold selected on CV |
| **XGB at tuned threshold 0.32** | **0.632** | **0.846** | **best model** |

At the best model's tuned threshold: Precision equals 0.542, Recall equals
0.757. The model catches 283 actual churners out of 374 in the test set, and
raises 239 false alarms out of 1,035 non-churners.

---

# 3. Guiding questions

## 3.1 The Accuracy trap

Accuracy on the locked test set is 0.766, only slightly above the score of
0.735 that a constant "not churning" model would receive - and so by this
single number the model looks generic and unimpressive.

The Confusion Matrix tells the real story: 91 errors of type FN, meaning 24
percent of real churners were missed, and 239 false alarms, which is an FP
rate of 23 percent on non-churners. Accuracy gives full credit to a huge
pile of easy True Negatives produced by the majority-class split, and almost
no weight to the 374 churners that I actually care about.

For LTV use, this is exactly the wrong trade.

## 3.2 Cost of errors

An FN error means a customer is lost: an LTV loss of many months of payment,
plus the cost of acquiring a substitute subscriber later on - that is
expensive.

An FP error means that a retention offer was sent to someone who would have
stayed regardless: discount given for nothing, representative time wasted. A
smaller cost, but not an absolute zero.

This is the reason that F1 was chosen and not Recall. Using Recall implicitly
assumes that FP is free, and that is not correct.

The threshold tuning then moves the decision point toward a higher Recall
(0.32 and not 0.5), which reflects the asymmetry in a principled way without
ignoring the opposite side of the monetary equation.

## 3.3 Is the model worth deploying?

The XGBoost model with a tuned threshold of 0.32 received an F1 of 0.632,
against the informative stratified baseline of 0.290. That is an improvement
of 0.342 points in absolute terms, or 118 percent in relative improvement.

The ROC-AUC of 0.846 confirms that there is a real ranking signal, which is
what the LTV consumer needs.

So yes, I would publish - but as a soft launch with reservations: in phase
one only on cases with high confidence (probability above 0.7).

In phase two, expand after three months of monitoring the Precision and
Recall scores, the drift of the distribution in a certain direction, and the
fairness gap relative to senior citizens.

The model is good enough to add value, but not good enough that I would let
it decide automatically on borderline cases.

## 3.4 What drives the predictions

The SHAP methods (on XGBoost) and Permutation Importance (model-agnostic,
measured against F1) agree on the top tier of features: `tenure`, `Contract`,
`InternetService`, `MonthlyCharges`, `PaymentMethod`, and `TotalCharges`.
These are consistent with the findings from the EDA: two-year contracts show
a churn rate of 2.8 percent, fiber-optic customers churn at a rate of 41.9
percent, and electronic-check payers churn at a rate of 45.3 percent. The
top-10 features by SHAP overlap with the top-10 LR L1 coefficients in 7 of
10 cases. One coefficient looked strange at first: in LR L1, the feature
`MonthlyCharges` received a negative coefficient, despite the fact that
churners pay more on average. The explanation is a multicollinearity
artifact, and not data leakage. The fiber-optic dummy absorbs the high-payment
signal (with a coefficient of plus 2.49), and `MonthlyCharges` becomes a
conditional correction. SHAP and Permutation, which both handle correlated
features correctly, rank `MonthlyCharges` as a positive driver - and this is
the correct answer. There is no real data leakage: the split is fair, all the
features are available before the churn decision is made, and the test set
did not touch any training step.

## 3.5 The five worst errors

The five wrong predictions with the highest confidence are all FN -

customers with long tenure (39 to 72 months), on contracts of one or two
years, who churned regardless. The model gave them churn probabilities
between 0.01 and 0.04 (meaning "definitely will stay"), and they nonetheless
left. These do not look like data quality issues or label noise - they look
like truly hard cases. The dataset stores customer attributes but not
reasons, and so the model has no signal for life events (a move, divorce,
downsizing) or competitor offers.

Beyond the obvious five, the systematic failure mode is the opposite: FP
errors on customers with monthly contracts, fiber optic, and electronic
payment (an FP rate of about 30 percent among them).

There the model sees the classic high-risk profile and over-predicts churn -
even on those who in the end stayed.

## 3.6 Stability

Standard deviation of F1 in CV across the five folds: 0.023 for XGBoost,
0.021 for RF, and 0.030 for LR L1. Stable within each model - but I would
not hand a stakeholder a single point estimate without the context around
it. The phrasing "0.587 plus or minus 0.023" is honest; the phrasing "0.587"
is a number that pretends to be more stable than it actually is.

The more interesting stability story is the train-test gap: for LR L1 the
gap is minus 0.002 (very stable), for XGBoost the gap is plus 0.049 (mild),
and for RF the gap is plus 0.131 (heavy overfit).

This gap is exactly the reason I deploy XGBoost and not RF, despite the two
of them being tied on test F1 at the tuned threshold. On slightly different
data, RF's performance is much less predictable.

## 3.7 Information leakage and time

The Telco dataset is a one-time cross-section without a timestamp column,
and therefore a Stratified 80/20 random split is the fair and correct split
here.

A time-based split is not even possible. I made sure that the test set did
not leak into any training step: the `ColumnTransformer` was fitted on
training only, all hyperparameter tuning grids went through CV on the
training sets, the optimal F1 threshold was chosen on out-of-fold predictions
from training only, and I touched the test set only once at the final
evaluation.

Had there been a time column, I would expect a random split to slightly
inflate F1 on the test compared to a split that respects time, because a
random split allows the model to see contemporaneous patterns that "leak
from the future". A time-based split would be a more honest measurement of
how the model performs on customers of the next quarter.

## 3.8 Production mindset

Four things to monitor on a weekly basis if the model goes live:

1. Precision and Recall on the most recent labeled cross-section, the moment
   churn outcomes are available.
2. Drift in features, using the Kolmogorov-Smirnov test on the `tenure` and
   `MonthlyCharges` columns, on the `Contract` mix, and on `PaymentMethod`.
3. The fairness gap among senior citizens (FP rate by `SeniorCitizen`).
   Currently the gap is about 9 percentage points wider for seniors, and it
   must not grow.
4. Drift in the distribution of the `predict_proba` output itself, which is
   often the earliest signal of model staleness.

Retraining triggers, if each of them persists for two weeks: an F1 drop of
more than 0.05, a fairness gap of more than 12 percentage points, or the
appearance of new categories of contract or payment method. The first
follow-up I would do regardless of any trigger is to calibrate the
probabilities (using `CalibratedClassifierCV` or isotonic regression on an
isolated calibration fold). Currently these are raw XGBoost outputs, and the
LTV consumer downstream needs real probabilities, not just rankings.

---

# 4. Model Card

## 4.1 Overview

| Field | Value |
|---|---|
| Owner | Orarr2 |
| Task | Binary classification, predicting whether a Telco subscriber will churn |
| Business framing | Per-customer churn probability, as input to the downstream LTV model |
| Dataset | IBM Telco Customer Churn from Kaggle, 7,043 rows, 21 columns, one-time cross-section |
| Target variable | 1 if Churn equals Yes, otherwise 0. Positive class rate: 26.5 percent |
| Split | Stratified 80/20 (5,634 training, 1,409 test) |

## 4.2 Metric and performance

The primary metric is F1 (balances FN and FP, both of which have a real
cost). The secondary metric is ROC-AUC (ranking quality for the LTV
consumer).

Baseline results on the test:

- baseline most_frequent: F1 equals 0.000 (empty bar).
- baseline stratified: F1 equals 0.290 (informative bar).

Performance of the best model (XGBoost, tuned, threshold 0.32) on the test:

- F1: 0.632
- Precision: 0.542
- Recall: 0.757
- ROC-AUC: 0.846
- PR-AUC: 0.660

Improvement vs the informative baseline: plus 0.342 F1 points, or plus 118
percent relative.

## 4.3 What the model leans on

The leading features, according to the agreement between SHAP and
Permutation Importance: `tenure`, `Contract`, `InternetService`,
`MonthlyCharges`, `PaymentMethod`, `TotalCharges`.

The linear LR L1 model, after tuning C to 10, zeroed exactly 0 features.
The "L1 produced sparsity" story does not hold here - with the optimal C,
the model retains all the features.

A suspicious finding, multicollinearity and not data leakage: the LR L1
coefficient on `MonthlyCharges` is negative, despite the fact that churners
pay more on average. The fiber-optic dummy absorbs the signal. SHAP and
Permutation, the right tools in such a situation, rank `MonthlyCharges` as a
positive driver.

There is no real data leakage. The split is fair, and there is no future
information in the training.

## 4.4 Limitations and failure modes

The five worst errors are all FN, on customers with long tenure and
multi-year contracts who churned regardless. The dataset does not capture
the reasons (life events, competitor offers).

A systematic FP on customers with a monthly contract, fiber optic, and
electronic-check payment: the model over-predicts churn for the classic
high-risk profile (an FP rate of about 30 percent).

Standard deviation of F1 in CV: 0.023. Reasonable, but it is always
necessary to report the band, and not a single point.

Train-test gap: for XGB the gap is plus 0.049 (mild). For RF the gap was
plus 0.131. Therefore RF was not deployed.

## 4.5 Fairness

Gender: an identical error rate of 23.4 percent for both genders. There is
no gender gap.

Senior citizens: an error rate of 31 percent versus 22 percent among
non-seniors. The gap stems from a higher FP rate among seniors. A fairness
flag is documented. It must be monitored in production.

## 4.6 Real-world deployment

The recommendation is a gradual launch with reservations:

1. Phase one: deployment only on cases with high confidence (probability
   above 0.7).
2. Phase two: expand after three months of monitoring on Precision, Recall,
   checking which way the data tends, and the FP gap among seniors.

Weekly monitoring: Precision, Recall, drift of the data in distribution of
`tenure`, `MonthlyCharges`, `Contract`, and `PaymentMethod`, and the FP gap
among seniors.

Retraining trigger, if it persists for two weeks: an F1 drop of more than
0.05, a fairness gap of more than 12 percentage points, or the appearance of
new categories of contract or payment method.

Next iteration: calibrate the probabilities for the LTV consumer (using
`CalibratedClassifierCV` or isotonic regression), investigate the FP gap
among seniors, and add customer-service call counts and complaint history if
they are accessible.

---

# 5. Reflection

The thing that surprised me most was that LR L1 won the cross-validation
with an F1 of 0.598, but lost the test set to the ensembles. I almost
shipped LR L1 based on the CV result alone. The explanation was revealed
when I looked at the gap between train and test: RF showed plus 0.131 (heavy
overfit), XGBoost showed plus 0.049 (mild), and LR L1 showed essentially
zero. The CV flattered the ensembles because each fold is smaller and the
overfit gets baked into the score. The test exposed this. Without doing the
investigation in Part 4, I would have written the wrong story.

The second surprise was that LR L1 did not actually produce sparsity. I
chose it for a story of interpretability via sparsity, and expected the
regularizer to zero out the weak features.

The CV grid chose C equals 10, which is the high end of the grid and almost
no regularization, and it zeroed exactly 0 features.

I left this in the report instead of hiding it.

The related finding about the negative coefficient on `MonthlyCharges` (a
multicollinearity component, and not noise) was a useful reminder: SHAP and
Permutation are the right tools for the question "what drives the
prediction" when features are correlated with each other.
