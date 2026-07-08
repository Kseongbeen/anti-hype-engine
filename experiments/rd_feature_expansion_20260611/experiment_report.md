# R&D Feature Expansion Experiment

## Added Features
- `feat_rd_absence_3y`: 최근 3개년 R&D 투자 공백 위험
- `feat_rd_to_assets_rank_gap`: 자산 대비 R&D 투자 부족 위험
- `feat_rd_sales_growth_mismatch`: 매출 성장 대비 R&D 성장 부진 위험
- `feat_rd_pipeline_disclosure_absence`: R&D 성과·파이프라인 공시 부재 위험
- `feat_rd_milestone_correction_ratio`: R&D 성과 공시 정정 비율 위험

## Cross Validation Metrics (2024 training frame)

| Model | ROC-AUC | Accuracy | F1 |
|---|---:|---:|---:|
| repaired baseline | 0.7463 | 0.6853 | 0.3395 |
| R&D expanded | 0.7498 | 0.6623 | 0.2843 |

## Added Feature Importances

| Feature | Importance |
|---|---:|
| `feat_rd_to_assets_rank_gap` | 3.9813% |
| `feat_rd_sales_growth_mismatch` | 2.0280% |
| `feat_rd_pipeline_disclosure_absence` | 0.9730% |
| `feat_rd_absence_3y` | 0.0000% |
| `feat_rd_milestone_correction_ratio` | 0.0000% |

## Total R&D Axis RF Weight

- R&D feature group total: 11.0751%

## Research / Data Rationale

- OpenDART's full financial statement API exposes fiscal year, report code, statement type, account name, current amount, cumulative amount, and related XBRL account fields, so R&D amount, assets, revenue, and future development-cost/intangible-asset accounts are appropriate DART-derived candidates.
- OpenDART's XBRL raw file API can be used later to parse detailed notes such as development costs and intangible assets when the project adds a broader DART ingestion step.
- Prior pharmaceutical R&D literature points to declining internal research output and greater reliance on external collaboration, so R&D amount alone is not enough; a small disclosure-based milestone signal was added as a substance check.
- Innovation literature commonly uses R&D intensity, but patent/output signals can be noisy or signaling-oriented in early-stage/high-tech firms, so this experiment keeps patent-like disclosure signals as low-weight auxiliary features instead of core score drivers.

## Notes

- This experiment uses the repaired experiment as its base and does not overwrite production CSV files.
- The added features are derived from DART-style financial statement fields and local disclosure text already stored in the project.
- `feat_rd_spending_decline_2y` was tested as a candidate but excluded from the selected experiment because it lowered cross-validation performance.
- Text disclosure features are deliberately compressed to two features so the R&D axis does not become a news/disclosure-frequency model.