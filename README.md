# LightGBM Alzheimer's Disease Classification Pipeline

A heavy, highly-optimized LightGBM classification pipeline for predicting Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) status using CSF biomarkers and cognitive scores.

## Features

- **Heavy LightGBM Model** with aggressive hyperparameters
- **Optuna Hyperparameter Optimization** (60 trials, 5-fold CV)
- **Comprehensive Evaluation Metrics** including ROC-AUC, sensitivity, specificity
- **Advanced Visualizations**: Confusion matrix, feature importance, SHAP plots, ROC curves
- **GPU Support** (automatically detected)
- **Production-Ready** code structure

## Dataset Requirements

### Input Format
- File type: Excel (.xlsx)
- Target column: `diagnosis` (values: AD, MCI, CN)

### Required Features
```python
FEATURE_COLUMNS = [
    'csf_abeta40_value',    # CSF Amyloid-beta 40
    'csf_abeta42_value',    # CSF Amyloid-beta 42
    'csf_nfl_value',        # CSF Neurofilament light chain
    'csf_ptau_value',       # CSF Phosphorylated tau
    'csf_tau_value',        # CSF Total tau
    'formic_acid_value',    # Formic acid
    'lactoferrin_value',    # Lactoferrin
    'dha_value',            # Docosahexaenoic acid
    'ntp_value',            # NTP value
    'mmse_score'            # Mini-Mental State Examination score
]
```

## Installation

### Local Environment
```bash
pip install -r requirements.txt
```

### Google Colab
```python
!pip install lightgbm optuna shap pandas openpyxl scikit-learn matplotlib seaborn
```

## Usage

### Google Colab (Recommended)

1. **Upload the script to Colab:**
   - Upload `lightgbm_ad_classifier.py` to your Colab session

2. **Upload your data:**
   ```python
   from google.colab import files
   uploaded = files.upload()
   data_file = list(uploaded.keys())[0]
   ```

3. **Run the pipeline:**
   ```python
   %run lightgbm_ad_classifier.py
   model, metrics = main(data_file)
   ```

### Local Execution

```python
from lightgbm_ad_classifier import main

# Run the pipeline
model, metrics = main('path/to/your/data.xlsx')
```

### Python Script
```python
python lightgbm_ad_classifier.py
```

## Pipeline Steps

1. **Data Loading & Preprocessing**
   - Loads Excel file
   - Selects features and target
   - Drops missing values
   - Encodes labels
   - 80/20 stratified train-test split

2. **Hyperparameter Optimization**
   - Optuna with TPE sampler
   - 60 trials
   - 5-fold stratified cross-validation
   - Search space:
     - `num_leaves`: 31-255
     - `max_depth`: 12-30
     - `learning_rate`: 0.001-0.05
     - `n_estimators`: 500-3000
     - `min_child_samples`: 10-60
     - L1/L2 regularization
     - Feature/bagging fractions

3. **Model Training**
   - Trains LightGBM with best parameters
   - GPU acceleration if available

4. **Comprehensive Evaluation**
   - Accuracy, precision, recall, F1-score
   - Per-class metrics
   - Confusion matrix
   - ROC-AUC (micro, macro, weighted)
   - Sensitivity & specificity

5. **Visualizations**
   - Confusion matrix heatmap
   - Feature importance plot
   - SHAP summary plot
   - ROC curves (one-vs-rest)

6. **Results Export**
   - `results/performance_metrics.txt`
   - `results/classification_report.txt`
   - `results/feature_importance.csv`
   - `results/confusion_matrix.png`
   - `results/feature_importance.png`
   - `results/shap_summary.png`
   - `results/roc_curves.png`

## Output Format

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

--- Confusion Matrix ---
           AD     CN    MCI
    AD     28      1      1
    CN      2     40      8
   MCI      1     10     39

--- ROC-AUC Metrics ---
AD AUC: 0.98
CN AUC: 0.92
MCI AUC: 0.90
Micro-average AUC: 0.93
Macro-average AUC: 0.93
Weighted-average AUC: 0.93

--- Sensitivity & Specificity per Class ---
AD: Sensitivity 0.93 | Specificity 0.97
CN: Sensitivity 0.80 | Specificity 0.88
MCI: Sensitivity 0.77 | Specificity 0.91

--- Cross-Validation Summary ---
Mean CV Accuracy: 0.8234

--- Best Parameters ---
  num_leaves: 127
  max_depth: 18
  learning_rate: 0.0123
  n_estimators: 1500
  lambda_l1: 0.245
  lambda_l2: 1.234
  feature_fraction: 0.85
  bagging_fraction: 0.82
======================================================================
```

## Configuration

Edit these constants in `lightgbm_ad_classifier.py`:

```python
RANDOM_SEED = 42          # Random seed for reproducibility
TEST_SIZE = 0.2           # Test set size (20%)
N_OPTUNA_TRIALS = 60      # Number of Optuna trials
CV_FOLDS = 5              # Cross-validation folds
RESULTS_DIR = './results' # Output directory
```

## Hardware Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4+ CPU cores
- **GPU**: Optional but recommended (CUDA-compatible)

## Performance Tips

1. **GPU Acceleration**: If you have a GPU, LightGBM will automatically use it
2. **Parallel Processing**: LightGBM uses all available CPU cores by default
3. **Memory**: Larger `num_leaves` and `n_estimators` require more memory
4. **Speed vs Accuracy**: Reduce `N_OPTUNA_TRIALS` for faster results

## Troubleshooting

### Common Issues

**ImportError: No module named 'lightgbm'**
```bash
pip install lightgbm
```

**GPU not detected**
- Install CUDA toolkit
- Install GPU version: `pip install lightgbm[gpu]`

**Out of memory**
- Reduce `n_estimators` or `num_leaves`
- Reduce `N_OPTUNA_TRIALS`

**Missing columns in dataset**
- Ensure all 10 feature columns exist in your Excel file
- Check spelling and capitalization

## Model Performance

Expected performance on balanced datasets:
- **Accuracy**: 75-90%
- **Macro F1**: 0.75-0.88
- **ROC-AUC**: 0.85-0.95

Performance depends on:
- Dataset size
- Class balance
- Feature quality
- Hyperparameter tuning

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{lightgbm_ad_classifier,
  title={LightGBM Alzheimer's Disease Classification Pipeline},
  author={ML Pipeline Team},
  year={2025},
  url={https://github.com/your-repo/ML_06_ALZHEIMERS_LIGHTGBM}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For issues and questions, please open an issue on GitHub.

## Acknowledgments

- LightGBM: https://github.com/microsoft/LightGBM
- Optuna: https://optuna.org/
- SHAP: https://github.com/slundberg/shap
