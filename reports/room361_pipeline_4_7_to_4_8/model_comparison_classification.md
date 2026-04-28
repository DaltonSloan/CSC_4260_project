# Classification Model Comparison

Band metrics are limited by sparse labels and low class diversity in some folds.

| model_name                        | anchor_eval_windows | anchor_band_accuracy | anchor_mean_band_distance | anchor_mean_confidence | diagnostic_splits | diagnostic_accuracy | diagnostic_mean_band_distance |
| --------------------------------- | ------------------- | -------------------- | ------------------------- | ---------------------- | ----------------- | ------------------- | ----------------------------- |
| gradient_boosting_band_classifier | 3                   | 0.6667               | 0.3333                    | 1                      | 2                 | 0.5                 | 0.5                           |
| logistic_band_classifier          | 3                   | 0.6667               | 0.3333                    | 0.9937                 | 2                 | 0.5                 | 0.5                           |
| physics_baseline_band             | 3                   | 0.6667               | 0.3333                    |                        | 2                 | 1                   | 0                             |
| random_forest_band_classifier     | 3                   | 0.6667               | 0.3333                    | 0.8723                 | 2                 | 0.5                 | 0.5                           |
