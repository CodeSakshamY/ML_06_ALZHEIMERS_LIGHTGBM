# Quick Start Guide

Get started with the LightGBM Alzheimer's Disease Classification Pipeline in 5 minutes!

## Option 1: Google Colab (Easiest)

1. **Open Google Colab**: https://colab.research.google.com/

2. **Upload the notebook**:
   - Click "File" → "Upload notebook"
   - Upload `lightgbm_ad_classifier_colab.ipynb`

3. **Run all cells**:
   - Click "Runtime" → "Run all"
   - When prompted, upload your Excel data file

4. **Download results**:
   - Results will be automatically downloaded as a zip file

**Total time**: ~10-15 minutes (depending on data size)

---

## Option 2: Local Python

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

Ensure your Excel file (.xlsx) has these columns:
- `csf_abeta40_value`
- `csf_abeta42_value`
- `csf_nfl_value`
- `csf_ptau_value`
- `csf_tau_value`
- `formic_acid_value`
- `lactoferrin_value`
- `dha_value`
- `ntp_value`
- `mmse_score`
- `diagnosis` (values: AD, MCI, CN)

See `example_data_template.csv` for reference.

### Step 3: Run the Pipeline

```python
from lightgbm_ad_classifier import main

# Run with your data file
model, metrics = main('path/to/your/data.xlsx')
```

Or run as a script:

```bash
python lightgbm_ad_classifier.py
```

Then edit the file to uncomment and set your data path:
```python
if __name__ == "__main__":
    data_file_path = "your_data.xlsx"
    model, metrics = main(data_file_path)
```

### Step 4: Check Results

Results are saved in `./results/`:
- `performance_metrics.txt` - All metrics
- `classification_report.txt` - Sklearn classification report
- `confusion_matrix.png` - Confusion matrix heatmap
- `feature_importance.png` - Feature importance plot
- `shap_summary.png` - SHAP explainability plot
- `roc_curves.png` - ROC curves for all classes
- `feature_importance.csv` - Feature importance data

---

## Option 3: Command Line (Advanced)

```bash
# Install dependencies
pip install lightgbm optuna shap pandas openpyxl scikit-learn matplotlib seaborn

# Run in Python
python -c "from lightgbm_ad_classifier import main; main('data.xlsx')"
```

---

## Expected Output

```
======================================================================
### Performance Metrics
======================================================================

Accuracy: 0.85
Macro Avg → Precision: 0.84 | Recall: 0.83 | F1: 0.83
Weighted Avg → Precision: 0.85 | Recall: 0.85 | F1: 0.85

--- Per-Class ---
AD: Precision 0.95 | Recall 0.93 | F1 0.94
CN: Precision 0.78 | Recall 0.80 | F1 0.79
MCI: Precision 0.79 | Recall 0.77 | F1 0.78

...
======================================================================
```

---

## Troubleshooting

### "Missing columns in dataset"
- Check that your Excel file has all 10 feature columns + 1 target column
- Column names must match exactly (case-sensitive)

### "Out of memory"
- Reduce `N_OPTUNA_TRIALS` from 60 to 30 in the script
- Reduce `n_estimators` search range

### "Takes too long"
- Use fewer Optuna trials (edit `N_OPTUNA_TRIALS`)
- Use a smaller dataset for initial testing
- Enable GPU if available

### ImportError
```bash
pip install --upgrade lightgbm optuna shap
```

---

## Customization

Edit these parameters in the script:

```python
N_OPTUNA_TRIALS = 60      # Reduce to 30 for faster results
TEST_SIZE = 0.2           # Change train/test split
CV_FOLDS = 5              # Change cross-validation folds
RANDOM_SEED = 42          # Change for different random splits
```

---

## Next Steps

1. **Analyze Results**: Check `performance_metrics.txt` for detailed metrics
2. **Feature Importance**: Review which biomarkers are most predictive
3. **SHAP Values**: Understand model predictions with SHAP plots
4. **Hyperparameters**: Review best parameters in the output
5. **Iterate**: Adjust data preprocessing or feature engineering as needed

---

## Support

For issues, questions, or contributions:
- Check the full README.md for detailed documentation
- Open an issue on GitHub
- Review troubleshooting section

Happy modeling! 🧠🔬
