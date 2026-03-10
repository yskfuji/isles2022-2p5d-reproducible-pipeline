# ISLES 2.5D metrics comparison (2026-03-11)

## Validation

| Model | Threshold | Mean Dice | Median Dice | Precision | Recall | Detection | Lesion F1 (micro) | ASSD mm | HD95 mm | Abs volume diff mL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 | 0.85 | 0.7036 | 0.7620 | 0.8587 | 0.8126 | 0.9583 | 0.5128 | 2.1173 | 9.6535 | 2.5839 |
| v3 | 0.85 | 0.7126 | 0.7577 | 0.8305 | 0.8267 | 0.9583 | 0.5434 | 5.5034 | 15.2637 | 3.0504 |
| ensemble (best val config) | 0.75 | 0.7222 | 0.7622 | 0.8026 | 0.8649 | 0.9583 | 0.6151 | 7.5595 | 21.4002 | 2.9488 |

## Test

| Model | Threshold | Mean Dice | Median Dice | Precision | Recall | Detection | Lesion F1 (micro) | ASSD mm | HD95 mm | Abs volume diff mL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 | 0.90 | 0.5879 | 0.7099 | 0.8743 | 0.4493 | 0.9200 | 0.4159 | 7.6139 | 19.2883 | 18.5222 |
| v3 | 0.90 | 0.6243 | 0.6748 | 0.8916 | 0.4847 | 1.0000 | 0.4715 | 4.8634 | 18.0421 | 17.2420 |
| ensemble (val-fixed final) | 0.75 | 0.6212 | 0.6690 | 0.8557 | 0.5390 | 0.9600 | 0.5357 | 6.8904 | 21.4013 | 15.9499 |

## Notes

- The fair final ensemble test result is the val-selected configuration fixed on test: threshold 0.75, min_size 32, prob_filter 0.90.
- The single-model test thresholds above are each model's own selected operating point from the available evaluation runs.
- Lesion F1 is reported as micro-averaged lesion-wise F1 from connected-component matching.
- ASSD and HD95 are boundary distance metrics in millimeters; lower is better.
