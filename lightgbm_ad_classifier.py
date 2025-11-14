"""
LightGBM Classification Pipeline for Alzheimer's Disease Prediction
====================================================================
Classifies patients into: AD (Alzheimer's Disease), MCI (Mild Cognitive Impairment), CN (Cognitively Normal)

Author: ML Pipeline
Date: 2025-11-14
"""

# ============================================================================
# INSTALLATION (Run in Colab)
# ============================================================================
# !pip install lightgbm optuna shap pandas openpyxl scikit-learn matplotlib seaborn imbalanced-learn

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# LightGBM
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced-learn for SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optuna for hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# SHAP for feature importance
import shap

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_COLUMNS = [
    'csf_abeta40_value', 'csf_abeta42_value', 'csf_nfl_value', 'csf_ptau_value',
    'csf_tau_value', 'formic_acid_value', 'lactoferrin_value',
    'dha_value', 'ntp_value', 'mmse_score'
]

TARGET_COLUMN = 'diagnosis'
TEST_SIZE = 0.2
N_OPTUNA_TRIALS = 60
CV_FOLDS = 5
RESULTS_DIR = './results'

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_data(file_path):
    """
    Load Excel file, preprocess data, and split into train/test sets.

    Args:
        file_path (str): Path to Excel file

    Returns:
        tuple: X_train, X_test, y_train, y_test, label_encoder, feature_names
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load Excel file
    df = pd.read_excel(file_path)
    print(f"✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Select features and target
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    df_filtered = df[required_columns].copy()
    print(f"✓ Selected {len(FEATURE_COLUMNS)} features + 1 target column")

    # Drop rows with missing values
    initial_rows = len(df_filtered)
    df_filtered = df_filtered.dropna()
    dropped_rows = initial_rows - len(df_filtered)
    print(f"✓ Dropped {dropped_rows} rows with missing values ({len(df_filtered)} remaining)")

    # ========================================================================
    # FEATURE ENGINEERING: Add critical biomarker ratios
    # ========================================================================
    print(f"\n--- Feature Engineering ---")

    # Aβ42/Aβ40 ratio - Gold standard biomarker for AD
    # Lower ratio indicates AD pathology
    df_filtered['abeta42_40_ratio'] = df_filtered['csf_abeta42_value'] / (df_filtered['csf_abeta40_value'] + 1e-10)
    print("✓ Added Aβ42/Aβ40 ratio (gold standard AD biomarker)")

    # Tau/Aβ42 ratio - Discriminates AD from CN/MCI
    # Higher ratio indicates AD pathology
    df_filtered['tau_abeta42_ratio'] = df_filtered['csf_tau_value'] / (df_filtered['csf_abeta42_value'] + 1e-10)
    print("✓ Added Tau/Aβ42 ratio (discriminates AD from CN/MCI)")

    # pTau/Tau ratio - Indicates phosphorylation state
    # Helps distinguish MCI from CN
    df_filtered['ptau_tau_ratio'] = df_filtered['csf_ptau_value'] / (df_filtered['csf_tau_value'] + 1e-10)
    print("✓ Added pTau/Tau ratio (helps distinguish MCI from CN)")

    # pTau/Aβ42 ratio - Another important discriminator
    df_filtered['ptau_abeta42_ratio'] = df_filtered['csf_ptau_value'] / (df_filtered['csf_abeta42_value'] + 1e-10)
    print("✓ Added pTau/Aβ42 ratio")

    # Total biomarker burden (sum of pathological markers)
    df_filtered['biomarker_burden'] = (
        df_filtered['csf_tau_value'] +
        df_filtered['csf_ptau_value'] -
        df_filtered['csf_abeta42_value'] * 0.01  # Scale down Aβ42 as it's inversely related
    )
    print("✓ Added biomarker burden score")

    # Update feature list with engineered features
    engineered_features = [
        'abeta42_40_ratio', 'tau_abeta42_ratio', 'ptau_tau_ratio',
        'ptau_abeta42_ratio', 'biomarker_burden'
    ]
    all_features = FEATURE_COLUMNS + engineered_features

    # Check class distribution
    print(f"\n--- Target Distribution ---")
    class_counts = df_filtered[TARGET_COLUMN].value_counts()
    for label, count in class_counts.items():
        print(f"  {label}: {count} ({count/len(df_filtered)*100:.1f}%)")

    # Encode target labels
    X = df_filtered[all_features].values
    y = df_filtered[TARGET_COLUMN].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\n✓ Encoded labels: {dict(enumerate(label_encoder.classes_))}")

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )

    print(f"\n✓ Train-test split (stratified):")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"✓ Total features: {len(all_features)} ({len(FEATURE_COLUMNS)} original + {len(engineered_features)} engineered)")
    print("=" * 70)

    return X_train, X_test, y_train, y_test, label_encoder, all_features


# ============================================================================
# HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================================

def objective(trial, X_train, y_train, n_classes, class_weights_dict):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        n_classes: Number of classes
        class_weights_dict: Dictionary of class weights

    Returns:
        float: Mean cross-validation score
    """
    # Define hyperparameter search space (OPTIMIZED for MCI/CN separation)
    params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        'device': 'gpu' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
        'class_weight': class_weights_dict,  # Add class weighting
        'is_unbalance': True,  # Handle class imbalance

        # Tree structure - More conservative to prevent overfitting
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # Reduced from 31-255
        'max_depth': trial.suggest_int('max_depth', 5, 15),  # Reduced from 12-30

        # Learning parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=100),

        # Stronger regularization for MCI/CN separation
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),  # Increased from 10-60
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1.0, log=True),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 50.0, log=True),  # Stronger L1
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 50.0, log=True),  # Stronger L2

        # Sampling
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        # Advanced parameters
        'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 1.5),  # Require more gain to split
        'path_smooth': trial.suggest_float('path_smooth', 0.0, 1.0),
    }

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)

    # Cross-validation with SMOTE
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Apply SMOTE only on training fold
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)

        # Train and evaluate
        model.fit(X_train_resampled, y_train_resampled)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)

    return np.mean(scores)


def optimize_hyperparameters(X_train, y_train, n_classes):
    """
    Run Optuna hyperparameter optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        n_classes: Number of classes

    Returns:
        dict: Best hyperparameters, best CV score, class weights
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("=" * 70)

    # Compute class weights to handle imbalance
    print("\n--- Computing Class Weights ---")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    print("Class weights (to handle MCI/CN imbalance):")
    for class_idx, weight in class_weights_dict.items():
        print(f"  Class {class_idx}: {weight:.3f}")

    print(f"\nRunning {N_OPTUNA_TRIALS} trials with {CV_FOLDS}-fold cross-validation...")
    print("This may take several minutes...\n")

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_SEED)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, n_classes, class_weights_dict),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
        n_jobs=1  # LightGBM already uses parallelization
    )

    print("\n✓ Optimization complete!")
    print(f"  Best CV Accuracy: {study.best_value:.4f}")
    print(f"  Best Trial: #{study.best_trial.number}")

    # Construct best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        'device': 'gpu' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
        'class_weight': class_weights_dict,
        'is_unbalance': True,
    })

    print("\n--- Best Parameters ---")
    for key, value in sorted(best_params.items()):
        if key not in ['objective', 'num_class', 'metric', 'boosting_type', 'verbosity', 'random_state', 'device', 'class_weight', 'is_unbalance']:
            print(f"  {key}: {value}")
    print("=" * 70)

    return best_params, study.best_value


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_final_model(X_train, y_train, best_params):
    """
    Train final model with best hyperparameters and SMOTE.

    Args:
        X_train: Training features
        y_train: Training labels
        best_params: Best hyperparameters from Optuna

    Returns:
        LGBMClassifier: Trained model, resampled training data
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    # Apply SMOTE to balance classes (especially MCI/CN)
    print("\n--- Applying SMOTE for Class Balancing ---")
    print(f"Original training set size: {X_train.shape[0]}")
    print("Original class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"  Class {class_idx}: {count} samples")

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\nResampled training set size: {X_train_resampled.shape[0]}")
    print("Resampled class distribution:")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"  Class {class_idx}: {count} samples")

    # Train model on resampled data
    model = LGBMClassifier(**best_params)
    model.fit(X_train_resampled, y_train_resampled)

    print("\n✓ Model training complete!")
    print("=" * 70)

    return model, X_train_resampled


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_specificity(y_true, y_pred, n_classes):
    """
    Calculate specificity for each class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes

    Returns:
        dict: Specificity per class
    """
    cm = confusion_matrix(y_true, y_pred)
    specificity = {}

    for i in range(n_classes):
        # True Negatives: sum of all cells except row i and column i
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        # False Positives: sum of column i except diagonal
        fp = np.sum(cm[:, i]) - cm[i, i]

        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return specificity


def evaluate_model(model, X_test, y_test, label_encoder, best_cv_score, best_params):
    """
    Comprehensive model evaluation with all metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for class names
        best_cv_score: Best cross-validation score
        best_params: Best hyperparameters

    Returns:
        dict: All evaluation metrics
    """
    print("\n" + "=" * 70)
    print("### Performance Metrics")
    print("=" * 70)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Macro Avg → Precision: {precision_macro:.2f} | Recall: {recall_macro:.2f} | F1: {f1_macro:.2f}")
    print(f"Weighted Avg → Precision: {precision_weighted:.2f} | Recall: {recall_weighted:.2f} | F1: {f1_weighted:.2f}")

    # Per-class metrics
    print("\n--- Per-Class ---")
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    class_names = label_encoder.classes_
    specificity_per_class = calculate_specificity(y_test, y_pred, len(class_names))

    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision {precision_per_class[i]:.2f} | "
              f"Recall {recall_per_class[i]:.2f} | F1 {f1_per_class[i]:.2f}")

    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)

    # Print header
    header = "           " + "".join([f"{name:>7}" for name in class_names])
    print(header)

    # Print matrix
    for i, class_name in enumerate(class_names):
        row = f"    {class_name:>2}    " + "".join([f"{cm[i][j]:>7}" for j in range(len(class_names))])
        print(row)

    # ROC-AUC Metrics (One-vs-Rest)
    print("\n--- ROC-AUC Metrics ---")

    # Binarize labels for multiclass ROC-AUC
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))

    # Per-class AUC
    auc_per_class = {}
    for i, class_name in enumerate(class_names):
        auc_score = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        auc_per_class[class_name] = auc_score
        print(f"{class_name} AUC: {auc_score:.2f}")

    # Micro-average AUC
    auc_micro = roc_auc_score(y_test_bin, y_pred_proba, average='micro')
    print(f"Micro-average AUC: {auc_micro:.2f}")

    # Macro-average AUC
    auc_macro = roc_auc_score(y_test_bin, y_pred_proba, average='macro')
    print(f"Macro-average AUC: {auc_macro:.2f}")

    # Weighted-average AUC
    auc_weighted = roc_auc_score(y_test_bin, y_pred_proba, average='weighted')
    print(f"Weighted-average AUC: {auc_weighted:.2f}")

    # Specificity and Sensitivity
    print("\n--- Sensitivity & Specificity per Class ---")
    for i, class_name in enumerate(class_names):
        sensitivity = recall_per_class[i]  # Sensitivity = Recall
        specificity = specificity_per_class[i]
        print(f"{class_name}: Sensitivity {sensitivity:.2f} | Specificity {specificity:.2f}")

    # Cross-Validation Summary
    print("\n--- Cross-Validation Summary ---")
    print(f"Mean CV Accuracy: {best_cv_score:.4f}")

    print("\n--- Best Hyperparameters ---")
    important_params = ['num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                       'lambda_l1', 'lambda_l2', 'feature_fraction', 'bagging_fraction']
    for param in important_params:
        if param in best_params:
            print(f"  {param}: {best_params[param]}")

    print("=" * 70)

    # Return all metrics for saving
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'specificity_per_class': specificity_per_class,
        'confusion_matrix': cm,
        'auc_per_class': auc_per_class,
        'auc_micro': auc_micro,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'best_cv_score': best_cv_score,
        'best_params': best_params,
        'class_names': class_names,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {save_path}")


def plot_feature_importance(model, feature_names, save_path):
    """
    Plot and save feature importance.

    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        save_path: Path to save figure
    """
    # Get feature importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance (LightGBM)', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved feature importance: {save_path}")

    return importance_df


def plot_shap_summary(model, X_test, feature_names, save_path):
    """
    Plot and save SHAP summary plot.

    Args:
        model: Trained LightGBM model
        X_test: Test features
        feature_names: List of feature names
        save_path: Path to save figure
    """
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For multiclass, shap_values is a list of arrays
        # We'll plot the summary for all classes
        plt.figure(figsize=(12, 8))

        if isinstance(shap_values, list):
            # Multiclass case - plot mean absolute SHAP values
            shap_values_combined = np.abs(shap_values).mean(axis=0)
            shap.summary_plot(shap_values_combined, X_test,
                            feature_names=feature_names,
                            show=False)
        else:
            shap.summary_plot(shap_values, X_test,
                            feature_names=feature_names,
                            show=False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved SHAP summary plot: {save_path}")

    except Exception as e:
        print(f"⚠ Warning: Could not generate SHAP plot: {str(e)}")


def plot_roc_curves(metrics, save_path):
    """
    Plot and save ROC curves for all classes.

    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_test = metrics['y_test']
    y_pred_proba = metrics['y_pred_proba']
    class_names = metrics['class_names']
    n_classes = len(class_names)

    # Binarize labels
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Compute ROC curve for each class
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (class_name, color) in enumerate(zip(class_names, colors[:n_classes])):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})')

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curves: {save_path}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(metrics, importance_df, results_dir):
    """
    Save all results to files.

    Args:
        metrics: Dictionary of evaluation metrics
        importance_df: Feature importance dataframe
        results_dir: Directory to save results
    """
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    os.makedirs(results_dir, exist_ok=True)

    # Save performance metrics
    metrics_path = os.path.join(results_dir, 'performance_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("### Performance Metrics\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Accuracy: {metrics['accuracy']:.2f}\n")
        f.write(f"Macro Avg → Precision: {metrics['precision_macro']:.2f} | "
                f"Recall: {metrics['recall_macro']:.2f} | F1: {metrics['f1_macro']:.2f}\n")
        f.write(f"Weighted Avg → Precision: {metrics['precision_weighted']:.2f} | "
                f"Recall: {metrics['recall_weighted']:.2f} | F1: {metrics['f1_weighted']:.2f}\n\n")

        f.write("--- Per-Class ---\n")
        for i, class_name in enumerate(metrics['class_names']):
            f.write(f"{class_name}: Precision {metrics['precision_per_class'][i]:.2f} | "
                   f"Recall {metrics['recall_per_class'][i]:.2f} | "
                   f"F1 {metrics['f1_per_class'][i]:.2f}\n")

        f.write("\n--- Confusion Matrix ---\n")
        cm = metrics['confusion_matrix']
        class_names = metrics['class_names']
        header = "           " + "".join([f"{name:>7}" for name in class_names])
        f.write(header + "\n")
        for i, class_name in enumerate(class_names):
            row = f"    {class_name:>2}    " + "".join([f"{cm[i][j]:>7}" for j in range(len(class_names))])
            f.write(row + "\n")

        f.write("\n--- ROC-AUC Metrics ---\n")
        for class_name, auc_score in metrics['auc_per_class'].items():
            f.write(f"{class_name} AUC: {auc_score:.2f}\n")
        f.write(f"Micro-average AUC: {metrics['auc_micro']:.2f}\n")
        f.write(f"Macro-average AUC: {metrics['auc_macro']:.2f}\n")
        f.write(f"Weighted-average AUC: {metrics['auc_weighted']:.2f}\n")

        f.write("\n--- Sensitivity & Specificity per Class ---\n")
        for i, class_name in enumerate(metrics['class_names']):
            sensitivity = metrics['recall_per_class'][i]
            specificity = metrics['specificity_per_class'][i]
            f.write(f"{class_name}: Sensitivity {sensitivity:.2f} | Specificity {specificity:.2f}\n")

        f.write("\n--- Cross-Validation Summary ---\n")
        f.write(f"Mean CV Accuracy: {metrics['best_cv_score']:.4f}\n")

        f.write("\n--- Best Parameters ---\n")
        for key, value in sorted(metrics['best_params'].items()):
            if key not in ['objective', 'num_class', 'metric', 'boosting_type', 'verbosity', 'random_state', 'device']:
                f.write(f"  {key}: {value}\n")

        f.write("=" * 70 + "\n")

    print(f"✓ Saved performance metrics: {metrics_path}")

    # Save classification report
    report_path = os.path.join(results_dir, 'classification_report.txt')
    report = classification_report(metrics['y_test'], metrics['y_pred'],
                                   target_names=metrics['class_names'])
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)

    print(f"✓ Saved classification report: {report_path}")

    # Save feature importance
    importance_path = os.path.join(results_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Saved feature importance CSV: {importance_path}")

    print("=" * 70)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(data_file_path):
    """
    Main pipeline execution.

    Args:
        data_file_path (str): Path to Excel data file
    """
    print("\n" + "=" * 70)
    print(" LIGHTGBM ALZHEIMER'S DISEASE CLASSIFICATION PIPELINE")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder, feature_names = load_and_preprocess_data(data_file_path)
    n_classes = len(label_encoder.classes_)

    # 2. Hyperparameter optimization
    best_params, best_cv_score = optimize_hyperparameters(X_train, y_train, n_classes)

    # 3. Train final model with SMOTE
    model, X_train_resampled = train_final_model(X_train, y_train, best_params)

    # 4. Evaluate model
    metrics = evaluate_model(model, X_test, y_test, label_encoder, best_cv_score, best_params)

    # 5. Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Confusion Matrix
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], metrics['class_names'], cm_path)

    # Feature Importance
    fi_path = os.path.join(RESULTS_DIR, 'feature_importance.png')
    importance_df = plot_feature_importance(model, feature_names, fi_path)

    # SHAP Summary
    shap_path = os.path.join(RESULTS_DIR, 'shap_summary.png')
    plot_shap_summary(model, X_test, feature_names, shap_path)

    # ROC Curves
    roc_path = os.path.join(RESULTS_DIR, 'roc_curves.png')
    plot_roc_curves(metrics, roc_path)

    print("=" * 70)

    # 6. Save results
    save_results(metrics, importance_df, RESULTS_DIR)

    # Final summary
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✓ All results saved to: {RESULTS_DIR}/")
    print(f"✓ Test Accuracy: {metrics['accuracy']:.2f}")
    print(f"✓ Macro F1-Score: {metrics['f1_macro']:.2f}")
    print(f"✓ Macro AUC: {metrics['auc_macro']:.2f}")
    print("=" * 70 + "\n")

    return model, metrics


# ============================================================================
# GOOGLE COLAB EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    For Google Colab execution:

    1. Upload your Excel file using:
       from google.colab import files
       uploaded = files.upload()
       data_file = list(uploaded.keys())[0]

    2. Run the pipeline:
       model, metrics = main(data_file)
    """

    # Example: If running locally or with a specific file path
    # Uncomment and modify the path below:

    # data_file_path = "your_data.xlsx"
    # model, metrics = main(data_file_path)

    print("=" * 70)
    print("Ready to run!")
    print("=" * 70)
    print("\nFor Google Colab, use:")
    print("  from google.colab import files")
    print("  uploaded = files.upload()")
    print("  data_file = list(uploaded.keys())[0]")
    print("  model, metrics = main(data_file)")
    print("\nFor local execution:")
    print("  model, metrics = main('path/to/your/data.xlsx')")
    print("=" * 70)
