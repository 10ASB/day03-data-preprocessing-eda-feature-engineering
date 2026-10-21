# Day 3 — Data Preprocessing, EDA & Feature Engineering

**AI/ML Internship · Week 1 · Day 3**

This repository practices exploratory data analysis (EDA), data cleaning, missing-value handling, numerical/categorical preprocessing, visual analysis, and leakage-aware train/test workflows.

## Dataset

- `internB_delivery_delays.csv` — delivery-delay practice dataset (1,800 rows, 17 columns).
- Target column: `is_delayed`.
- `eda_summary.txt` — dataset summary and preprocessing notes.

## Files

| File | Purpose |
|---|---|
| `day_3.py` | EDA, cleaning, plotting, and preprocessing-pipeline practice |
| `internB_delivery_delays.csv` | Raw practice dataset |
| `eda_summary.txt` | EDA/preprocessing summary |

## Topics covered

- Inspect shape, data types, descriptive statistics, missing values, and duplicates
- Fill missing numerical values with medians and categorical values with modes
- Clean currency-formatted values and standardize text categories
- Univariate and bivariate analysis
- Correlation heatmaps
- `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, and `Pipeline`
- Train/test splitting and data-leakage awareness

## Setup and run

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
python day_3.py
```

Run from this repository folder so the script can find `internB_delivery_delays.csv`.

## Important learning notes

- Fit preprocessing steps on the training split only; then transform the test split.
- Avoid using information unavailable at prediction time. For delay prediction, fields such as `actual_delivery_days`, `customer_rating`, and `delay_complaint_raised` may reveal the outcome or be recorded after delivery. Review feature timing before training.
- Correlation indicates association, not causation.
- Inspect ranges and invalid values before treating columns as clean.

## Deliverable checklist

- [x] EDA script added
- [ ] Dataset and summary added
- [ ] Review invalid/outlier values and parse dates
- [ ] Document key findings from plots
- [ ] Confirm a leakage-safe feature set before model training
