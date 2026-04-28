# Physics Bias Diagnostics

This diagnostic checks where the airflow-aware CO2 physics baseline underpredicts the manual anchors and where the hybrid model corrects that bias.

- The largest physics miss occurred at 2026-04-07 14:38, where the smoothed physics estimate was 11.4 against a manual count of 33. The strongest physics response within the next 30 minutes reached 12.3 after about 7 minutes.
- Changing the outdoor CO2 assumption at that timestamp from 400 ppm to 430 ppm shifts the estimate only from 13.7 to 10.5, which is not enough to close the full gap by itself.
- Mean anchor-window flow near the worst miss was 1130.7 cfm, which supports the interpretation that active ventilation diluted the immediate CO2 signal.
- Across the available anchor windows, the hybrid model did not beat the physics baseline yet: the baseline MAE was 7.32 people versus 8.54 people for the hybrid model.

The interpretation should still be treated as preliminary because only three manual anchor windows are available.

## Diagnostic Table

| anchor_id | anchor_time         | true_count | physics_pred_at_anchor | physics_abs_error | best_model_pred_at_anchor | best_model_abs_error | residual_correction | co2_ppm_at_anchor | flow_cfm_at_anchor | flow_window_mean | physics_future_30m_peak | physics_future_30m_peak_time | physics_minutes_to_peak | physics_est_outdoor_400ppm | physics_est_outdoor_default | physics_est_outdoor_430ppm |
| --------- | ------------------- | ---------- | ---------------------- | ----------------- | ------------------------- | -------------------- | ------------------- | ----------------- | ------------------ | ---------------- | ----------------------- | ---------------------------- | ----------------------- | -------------------------- | --------------------------- | -------------------------- |
| A1        | 2026-04-07 14:38:00 | 33         | 11.3858                | 21.6142           | 11.3135                   | 21.6865              | -0.0724             | 528.75            | 1128.7064          | 1130.6938        | 12.2851                 | 2026-04-07 14:45:00          | 7                       | 13.7168                    | 12.1187                     | 10.5206                    |
| A2        | 2026-04-08 15:00:00 | 11         | 11.0219                | 0.0219            | 13.6567                   | 2.6567               | 2.6348              | 510.7143          | 1125.1082          | 1126.1323        | 13.3404                 | 2026-04-08 15:25:00          | 25                      | 11.7577                    | 10.1647                     | 8.5717                     |
| A3        | 2026-04-08 15:30:00 | 13         | 12.6792                | 0.3208            | 14.2681                   | 1.2681               | 1.5889              | 536.8571          | 1114.9987          | 1124.0825        | 15.4907                 | 2026-04-08 15:55:00          | 25                      | 14.4034                    | 12.8247                     | 11.2461                    |
