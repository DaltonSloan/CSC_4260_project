# Regression Model Comparison

Anchor-window metrics are the reportable accuracy numbers. Forward-chaining metrics are pseudo-label diagnostics only.

| model_name                 | anchor_eval_windows | anchor_mae | anchor_rmse | anchor_within_range_rate | anchor_mean_bias | diagnostic_splits | diagnostic_weighted_mae | diagnostic_weighted_rmse |
| -------------------------- | ------------------- | ---------- | ----------- | ------------------------ | ---------------- | ----------------- | ----------------------- | ------------------------ |
| physics_baseline           | 3                   | 7.3189     | 12.4803     | 0.6667                   | -7.3043          | 2                 | 0.368                   | 0.4492                   |
| gradient_boosting_residual | 3                   | 8.5371     | 12.6356     | 0.3333                   | -5.9206          | 2                 | 4.1774                  | 5.2273                   |
| random_forest_residual     | 3                   | 9.3909     | 12.7971     | 0                        | -5.0411          | 2                 | 3.9103                  | 4.0923                   |
| ridge_residual             | 3                   | 9.4902     | 12.2265     | 0                        | -7.3718          | 2                 | 2.7848                  | 4.1173                   |
