# HRRR lead-consistency experiment

Purpose: verify that the selected ±1-h same-valid-time positive definition
behaves consistently when the control lead is expanded from F12 to F07–F17.

Default development period:
- 2020-12-02 through 2025-09-23
- intentionally stops before the proposed final-test period

Sampling:
- both ±1-h positives must be available
- 250 accepted controls per meteorological season per control lead
- one random 256×256 patch per accepted control
- control patch must have >=1% of pixels >=1 mm / 6 h

Metrics:
- FSS: threshold 5 mm / 6 h, 15-grid-point neighborhood
- continuous spatial correlation
- ETS: threshold 5 mm / 6 h
- RMSE, bias-corrected RMSE, MAD, bias, means, maxima retained in CSV

Run:
    ./scripts/submit_lead_consistency_experiment.sh

After all 11 array tasks finish:
    python scripts/plot_lead_consistency_experiment.py \
      --experiment-dir "$DATA_DIR/derived/experiments/lead_consistency_v1"

