"""
Machine Learning Pipeline for Loan Default Classification

This module provides a comprehensive ML pipeline for loan default prediction
with robust overfitting prevention, hyperparameter optimization, and model evaluation.
"""

from typing import Dict, Tuple, Any, Optional, Union, List
import time
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Scikit-learn imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, RFECV, SelectFromModel
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, fbeta_score, precision_recall_curve, 
    roc_auc_score, precision_score, recall_score, make_scorer,
    classification_report, confusion_matrix
)
from sklearn.manifold import TSNE
from sklearn.impute import IterativeImputer
from sklearn.model_selection import (
    train_test_split, learning_curve, StratifiedKFold, 
    validation_curve, cross_val_score
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced-learn imports (keeping for potential future use)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Enable experimental features
from sklearn.experimental import enable_iterative_imputer

# Gradient boosting libraries
import xgboost as xgb
import lightgbm as lgb

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def fbeta_scorer(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
    """
    Custom F-beta scorer for Optuna's hyperparameter optimization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        beta: Beta parameter for F-beta score
        
    Returns:
        F-beta score
    """
    return fbeta_score(y_true, y_pred, beta=beta)


def evaluate_model_cv_with_overfitting_control(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: StratifiedKFold,
    fbeta_scorer_func: callable,
    beta: float,
    overfitting_penalty_weight: float = 0.5,
    max_overfitting_tolerance: float = 0.05
) -> Tuple[float, Dict[str, float]]:
    """
    Perform cross-validation with overfitting monitoring and penalty.
    
    Args:
        model: Machine learning model to evaluate
        X: Feature matrix
        y: Target vector
        cv_splits: Cross-validation splitter
        fbeta_scorer_func: F-beta scoring function
        beta: Beta parameter for F-beta score
        overfitting_penalty_weight: Weight for overfitting penalty
        max_overfitting_tolerance: Maximum acceptable overfitting gap
        
    Returns:
        Tuple of (penalized_score, detailed_metrics)
    """
    val_scores = []
    train_scores = []
    overfitting_gaps = []
    
    for train_idx, val_idx in cv_splits.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model on fold training data
        model.fit(X_fold_train, y_fold_train)

        # Evaluate on both training and validation folds
        y_train_pred = model.predict(X_fold_train)
        y_val_pred = model.predict(X_fold_val)
        
        train_score = fbeta_scorer_func(y_fold_train, y_train_pred)
        val_score = fbeta_scorer_func(y_fold_val, y_val_pred)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        overfitting_gaps.append(train_score - val_score)

    mean_val_score = np.mean(val_scores)
    mean_train_score = np.mean(train_scores)
    mean_overfitting_gap = np.mean(overfitting_gaps)
    
    # Apply penalty for overfitting
    overfitting_penalty = max(0, mean_overfitting_gap - max_overfitting_tolerance) * overfitting_penalty_weight
    penalized_score = mean_val_score - overfitting_penalty
    
    detailed_metrics = {
        'val_score': mean_val_score,
        'train_score': mean_train_score,
        'overfitting_gap': mean_overfitting_gap,
        'overfitting_penalty': overfitting_penalty,
        'penalized_score': penalized_score,
        'is_overfitting': mean_overfitting_gap > max_overfitting_tolerance
    }
    
    return penalized_score, detailed_metrics


def evaluate_model_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: StratifiedKFold,
    fbeta_scorer_func: callable,
    beta: float
) -> float:
    """
    Perform cross-validation using only real data with cost-sensitive learning.
    
    Uses class weights instead of synthetic sampling for better generalization
    to unseen test data.
    
    Args:
        model: Machine learning model to evaluate
        X: Feature matrix
        y: Target vector
        cv_splits: Cross-validation splitter
        fbeta_scorer_func: F-beta scoring function
        beta: Beta parameter for F-beta score
        
    Returns:
        Mean F-beta score across all CV folds
    """
    scores = []
    for train_idx, val_idx in cv_splits.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model on original data (using class weights for imbalance)
        model.fit(X_fold_train, y_fold_train)

        # Predict on validation data
        y_pred = model.predict(X_fold_val)
        scores.append(fbeta_scorer_func(y_fold_val, y_pred))

    return np.mean(scores)


def _initialize_pipeline_components(
    y_train: pd.Series,
    time_budget_minutes: int,
    beta: float,
    cv_folds: int
) -> Tuple[Dict[int, float], float, callable, StratifiedKFold, Dict, Dict, float, float]:
    """
    Initialize common components for the ML pipeline.
    
    Args:
        y_train: Training target vector
        time_budget_minutes: Total time budget for hyperparameter tuning
        beta: Beta parameter for F-beta score
        cv_folds: Number of cross-validation folds
        
    Returns:
        Tuple of initialized components
    """
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    scale_pos_weight = class_weights[1] / class_weights[0]

    print(f"Class distribution (Train): {np.bincount(y_train) / len(y_train)}")
    print(f"Class weights: {class_weight_dict}")
    print(f"Scale pos weight (for XGBoost): {scale_pos_weight:.3f}")
    print(f"Using F-beta score with beta={beta} (favoring recall)")

    # Create partial F-beta scorer
    fbeta_scorer_partial = lambda y_true, y_pred: fbeta_scorer(y_true, y_pred, beta)

    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Initialize result containers
    results = {}
    study_results = {}
    start_time = time.time()
    time_per_model = time_budget_minutes / 3

    print(f"\\nTime budget: {time_budget_minutes} minutes ({time_per_model:.1f} min per model)")
    print("Using cost-sensitive learning instead of synthetic sampling for better generalization")

    return (
        class_weight_dict, scale_pos_weight, fbeta_scorer_partial,
        cv, results, study_results, start_time, time_per_model
    )


def _optimize_model(
    model_name: str,
    objective_func: callable,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    time_budget_seconds: int
) -> optuna.Study:
    """
    Perform hyperparameter optimization using Optuna.
    
    Args:
        model_name: Name of the model being optimized
        objective_func: Optuna objective function
        X_train: Training features
        y_train: Training targets
        cv: Cross-validation splitter
        smote: SMOTE sampler
        time_budget_seconds: Time budget in seconds
        
    Returns:
        Optuna study object with optimization results
    """
    print(f"\\n{'=' * 20} {model_name.upper()} OPTIMIZATION {'=' * 20}")
    
    # Create study with better configuration
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,  # More random trials before TPE kicks in
            n_ei_candidates=24    # Better exploration-exploitation balance
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3
        )
    )
    
    study.optimize(
        objective_func, 
        timeout=time_budget_seconds,
        n_jobs=1,  # Ensure thread safety with sklearn
        show_progress_bar=True,
        # callbacks=[lambda study, trial: print(f"Trial {trial.number}: {trial.value:.4f}") if trial.number % 10 == 0 else None]
    )
    
    print(f"{model_name} Best F-beta: {study.best_value:.4f} (from {len(study.trials)} trials)")
    return study


def _train_models(
    models_dict: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> None:
    """
    Train models on the original training set using cost-sensitive learning.
    
    Args:
        models_dict: Dictionary of model names and instances
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
    """
    print(f"\\n{'=' * 20} MODEL TRAINING {'=' * 20}")
    
    for name, model in models_dict.items():
        print(f"Training {name} on original data (size: {len(X_train)})")
        
        if name == 'XGBoost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        elif name == 'LightGBM':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)]
            )
        else:
            model.fit(X_train, y_train)


def _evaluate_models_on_validation(
    models_dict: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    beta: float,
    results_dict: Dict,
    study_results_dict: Dict
) -> None:
    """
    Evaluate trained models on validation set.
    
    Args:
        models_dict: Dictionary of trained models
        X_val: Validation features
        y_val: Validation targets
        beta: Beta parameter for F-beta score
        results_dict: Dictionary to store results
        study_results_dict: Dictionary of study results
    """
    print(f"\\n{'=' * 20} MODEL EVALUATION {'=' * 20}")
    print("Evaluating models on independent Validation Set for model selection.")

    for name, model in models_dict.items():
        print(f"\\nEvaluating {name} on validation set...")
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        val_fbeta = fbeta_score(y_val, y_val_pred, beta=beta)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_proba)

        # Check for overfitting
        cv_fbeta = study_results_dict[name].best_value
        overfitting_score = cv_fbeta - val_fbeta

        # Store results
        results_dict[name] = {
            'model': model,
            'cv_fbeta': cv_fbeta,
            'val_fbeta': val_fbeta,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_auc': val_auc,
            'overfitting': overfitting_score,
            'y_val_pred': y_val_pred,
            'y_val_proba': y_val_proba
        }

        # Print results
        print(f"  CV F-beta: {cv_fbeta:.4f}")
        print(f"  Validation F-beta: {val_fbeta:.4f}")
        print(f"  Validation Precision: {val_precision:.4f}")
        print(f"  Validation Recall: {val_recall:.4f}")
        print(f"  Validation AUC: {val_auc:.4f}")
        overfitting_warning = '⚠️' if overfitting_score > 0.05 else '✅'
        print(f"  Overfitting (CV - Val F-beta): {overfitting_score:.4f} {overfitting_warning}")


def _select_and_ensemble_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    results_dict: Dict,
    study_results_dict: Dict,
    beta: float
) -> Tuple[VotingClassifier, Dict, str, Any, float, float]:
    """
    Select best models and create ensemble.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets
        smote: SMOTE sampler
        results_dict: Model evaluation results
        study_results_dict: Hyperparameter optimization results
        beta: Beta parameter for F-beta score
        
    Returns:
        Tuple of ensemble model, retrained models, best model name, 
        best model object, ensemble validation F-beta, and AUC
    """
    print(f"\\n{'=' * 20} MODEL SELECTION AND ENSEMBLE CREATION {'=' * 20}")

    # Calculate adjusted scores with overfitting penalty
    model_scores = {}
    for name, result in results_dict.items():
        overfitting_penalty = max(0, result['overfitting'] - 0.01) * 2.0
        adjusted_score = result['val_fbeta'] - overfitting_penalty
        model_scores[name] = adjusted_score
        print(f"{name}: Val F-beta={result['val_fbeta']:.4f}, "
              f"Overfitting={result['overfitting']:.4f}, "
              f"Adjusted Score={adjusted_score:.4f}")

    # Select top 2 models for ensemble
    best_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    print(f"\\nSelected for ensemble: {best_models[0][0]} and {best_models[1][0]}")

    # Combine train and validation sets
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    print(f"Combined Train+Val set: {X_trainval.shape}")

    # Retrain selected models
    retrained_models = {}
    for name, _ in best_models:
        if name == 'RandomForest':
            model = RandomForestClassifier(
                **study_results_dict['RandomForest'].best_params,
                random_state=42, n_jobs=-1
            )
        elif name == 'XGBoost':
            model = xgb.XGBClassifier(
                **study_results_dict['XGBoost'].best_params,
                random_state=42
            )
        elif name == 'LightGBM':
            model = lgb.LGBMClassifier(
                **study_results_dict['LightGBM'].best_params,
                random_state=42
            )

        print(f"Retraining {name} on combined train+val data...")
        model.fit(X_trainval, y_trainval)
        retrained_models[name] = model

    # Create ensemble
    ensemble_estimators = [(name, retrained_models[name]) for name, _ in best_models]
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft', n_jobs=-1)

    print(f"Training Ensemble with {', '.join([name for name, _ in ensemble_estimators])}...")
    ensemble.fit(X_trainval, y_trainval)

    # Evaluate ensemble on validation set
    ensemble_val_pred = ensemble.predict(X_val)
    ensemble_val_proba = ensemble.predict_proba(X_val)[:, 1]

    ensemble_val_fbeta = fbeta_score(y_val, ensemble_val_pred, beta=beta)
    ensemble_val_auc = roc_auc_score(y_val, ensemble_val_proba)
    ensemble_val_precision = precision_score(y_val, ensemble_val_pred)
    ensemble_val_recall = recall_score(y_val, ensemble_val_pred)

    print(f"\\nEnsemble Validation Performance:")
    print(f"  F-beta: {ensemble_val_fbeta:.4f}")
    print(f"  Precision: {ensemble_val_precision:.4f}")
    print(f"  Recall: {ensemble_val_recall:.4f}")
    print(f"  AUC: {ensemble_val_auc:.4f}")

    best_single_model_name = best_models[0][0]
    best_model_obj = retrained_models[best_single_model_name]

    return (
        ensemble, retrained_models, best_single_model_name,
        best_model_obj, ensemble_val_fbeta, ensemble_val_auc
    )


def _final_test_evaluation(
    best_model_obj: Any,
    ensemble: VotingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    beta: float,
    best_single_model_name: str
) -> Tuple[float, float, float, float, float, float]:
    """
    Perform final evaluation on test set.
    
    Args:
        best_model_obj: Best single model
        ensemble: Ensemble model
        X_test: Test features
        y_test: Test targets
        beta: Beta parameter for F-beta score
        best_single_model_name: Name of best single model
        
    Returns:
        Tuple of test metrics for best single model and ensemble
    """
    print(f"\\n{'=' * 20} FINAL TEST SET EVALUATION {'=' * 20}")
    print("⚠️  Final, unbiased evaluation on unseen test data.")

    # Evaluate best single model
    y_test_pred_best = best_model_obj.predict(X_test)
    y_test_proba_best = best_model_obj.predict_proba(X_test)[:, 1]

    test_fbeta_best = fbeta_score(y_test, y_test_pred_best, beta=beta)
    test_auc_best = roc_auc_score(y_test, y_test_proba_best)
    test_precision_best = precision_score(y_test, y_test_pred_best)
    test_recall_best = recall_score(y_test, y_test_pred_best)

    # Evaluate ensemble
    ensemble_test_pred = ensemble.predict(X_test)
    ensemble_test_proba = ensemble.predict_proba(X_test)[:, 1]

    ensemble_test_fbeta = fbeta_score(y_test, ensemble_test_pred, beta=beta)
    ensemble_test_auc = roc_auc_score(y_test, ensemble_test_proba)
    ensemble_test_precision = precision_score(y_test, ensemble_test_pred)
    ensemble_test_recall = recall_score(y_test, ensemble_test_pred)

    # Print results
    print(f"\\n🏆 BEST SINGLE MODEL ({best_single_model_name}) - TEST PERFORMANCE:")
    print(f"  F-beta: {test_fbeta_best:.4f}")
    print(f"  Precision: {test_precision_best:.4f}")
    print(f"  Recall: {test_recall_best:.4f}")
    print(f"  AUC: {test_auc_best:.4f}")

    print(f"\\n🏆 ENSEMBLE MODEL - TEST PERFORMANCE:")
    print(f"  F-beta: {ensemble_test_fbeta:.4f}")
    print(f"  Precision: {ensemble_test_precision:.4f}")
    print(f"  Recall: {ensemble_test_recall:.4f}")
    print(f"  AUC: {ensemble_test_auc:.4f}")

    return (
        test_fbeta_best, test_auc_best, test_precision_best, test_recall_best,
        ensemble_test_fbeta, ensemble_test_auc
    )

def create_overfitting_aware_ml_pipeline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    time_budget_minutes: int = 15,
    beta: float = 1.5,
    cv_folds: int = 3,
    overfitting_tolerance: float = 0.03,
    overfitting_penalty_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Enhanced ML pipeline with built-in overfitting prevention during hyperparameter optimization.
    
    This pipeline extends the original create_ml_pipeline by incorporating overfitting monitoring
    directly into the Optuna optimization process, ensuring models are prevented from overfitting
    during training rather than just detecting it afterwards.
    
    Args:
        X_train: Training feature matrix
        X_val: Validation feature matrix
        X_test: Test feature matrix
        y_train: Training target vector
        y_val: Validation target vector
        y_test: Test target vector
        time_budget_minutes: Time budget for hyperparameter tuning (minutes)
        beta: Beta parameter for F-beta score (>1 favors recall over precision)
        cv_folds: Number of cross-validation folds
        overfitting_tolerance: Maximum acceptable overfitting gap (train - val)
        overfitting_penalty_weight: Weight for overfitting penalty in optimization
        
    Returns:
        Dictionary containing all results, models, and evaluation metrics
    """
    print("=" * 70)
    print("OVERFITTING-AWARE ML PIPELINE - ENHANCED OPTIMIZATION")
    print("=" * 70)
    print(f"Overfitting tolerance: {overfitting_tolerance:.3f}")
    print(f"Overfitting penalty weight: {overfitting_penalty_weight:.1f}")

    # Initialize pipeline components
    (class_weight_dict, scale_pos_weight, fbeta_scorer_func,
     cv, results, study_results, start_time, time_per_model) = \
        _initialize_pipeline_components(y_train, time_budget_minutes, beta, cv_folds)

    # Enhanced objective functions with overfitting awareness
    def rf_objective_with_overfitting_control(trial):
        """Random Forest objective with overfitting penalty."""
        # Adjust regularization parameters based on trial history
        trial_number = trial.number
        regularization_boost = min(trial_number / 50, 0.5)  # Gradually increase regularization
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=25),
            'max_depth': trial.suggest_int('max_depth', 3, max(3, 8 - int(regularization_boost * 2))),
            'min_samples_split': trial.suggest_int('min_samples_split', 
                                                 max(10, 10 + int(regularization_boost * 20)), 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 
                                                max(5, 5 + int(regularization_boost * 10)), 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 
                                                       0.001 * (1 + regularization_boost), 0.02),
            'class_weight': 'balanced_subsample',
            'random_state': 42,
            'n_jobs': -1,
            'bootstrap': True,
            'oob_score': True
        }
        
        model = RandomForestClassifier(**params)
        penalized_score, metrics = evaluate_model_cv_with_overfitting_control(
            model, X_train, y_train, cv, fbeta_scorer_func, beta,
            overfitting_penalty_weight, overfitting_tolerance
        )
        
        # Store overfitting info in trial user attributes
        trial.set_user_attr('overfitting_gap', metrics['overfitting_gap'])
        trial.set_user_attr('is_overfitting', metrics['is_overfitting'])
        trial.set_user_attr('val_score', metrics['val_score'])
        
        return penalized_score

    def xgb_objective_with_overfitting_control(trial):
        """XGBoost objective with overfitting penalty."""
        trial_number = trial.number
        regularization_boost = min(trial_number / 50, 0.5)
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=25),
            'max_depth': trial.suggest_int('max_depth', 3, max(3, 6 - int(regularization_boost))),
            'learning_rate': trial.suggest_float('learning_rate', 
                                               0.01, max(0.01, 0.2 - regularization_boost * 0.1), log=True),
            'subsample': trial.suggest_float('subsample', 
                                           max(0.6, 0.7 - regularization_boost * 0.1), 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 
                                                  max(0.6, 0.7 - regularization_boost * 0.1), 0.9),
            'gamma': trial.suggest_float('gamma', 0.1 * (1 + regularization_boost), 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1 * (1 + regularization_boost), 20),
            'reg_lambda': trial.suggest_float('reg_lambda', 2 * (1 + regularization_boost), 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 
                                                max(3, 3 + int(regularization_boost * 5)), 15),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        model = xgb.XGBClassifier(**params)
        penalized_score, metrics = evaluate_model_cv_with_overfitting_control(
            model, X_train, y_train, cv, fbeta_scorer_func, beta,
            overfitting_penalty_weight, overfitting_tolerance
        )
        
        trial.set_user_attr('overfitting_gap', metrics['overfitting_gap'])
        trial.set_user_attr('is_overfitting', metrics['is_overfitting'])
        trial.set_user_attr('val_score', metrics['val_score'])
        
        return penalized_score

    def lgb_objective_with_overfitting_control(trial):
        """LightGBM objective with overfitting penalty."""
        trial_number = trial.number
        regularization_boost = min(trial_number / 50, 0.5)
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=25),
            'max_depth': trial.suggest_int('max_depth', 3, max(3, 6 - int(regularization_boost))),
            'learning_rate': trial.suggest_float('learning_rate', 
                                               0.01, max(0.01, 0.2 - regularization_boost * 0.1), log=True),
            'subsample': trial.suggest_float('subsample', 
                                           max(0.6, 0.7 - regularization_boost * 0.1), 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 
                                                  max(0.6, 0.7 - regularization_boost * 0.1), 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1 * (1 + regularization_boost), 20),
            'reg_lambda': trial.suggest_float('reg_lambda', 2 * (1 + regularization_boost), 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 
                                                 max(20, 20 + int(regularization_boost * 50)), 200),
            'num_leaves': trial.suggest_int('num_leaves', 
                                          max(10, 10 + int(regularization_boost * 5)), 50),
            'min_child_weight': trial.suggest_float('min_child_weight', 
                                                  1 * (1 + regularization_boost), 20),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        model = lgb.LGBMClassifier(**params)
        penalized_score, metrics = evaluate_model_cv_with_overfitting_control(
            model, X_train, y_train, cv, fbeta_scorer_func, beta,
            overfitting_penalty_weight, overfitting_tolerance
        )
        
        trial.set_user_attr('overfitting_gap', metrics['overfitting_gap'])
        trial.set_user_attr('is_overfitting', metrics['is_overfitting'])
        trial.set_user_attr('val_score', metrics['val_score'])
        
        return penalized_score

    # Perform overfitting-aware hyperparameter optimization
    print(f"\n🔍 Using overfitting-aware optimization with {cv_folds}-fold CV")
    
    xgb_study = _optimize_model('XGBoost', xgb_objective_with_overfitting_control, 
                               X_train, y_train, cv, time_per_model * 60)
    study_results['XGBoost'] = xgb_study

    lgb_study = _optimize_model('LightGBM', lgb_objective_with_overfitting_control, 
                               X_train, y_train, cv, time_per_model * 60)
    study_results['LightGBM'] = lgb_study

    rf_study = _optimize_model('RandomForest', rf_objective_with_overfitting_control, 
                              X_train, y_train, cv, time_per_model * 60)
    study_results['RandomForest'] = rf_study

    # Print overfitting analysis for best trials
    print(f"\n{'=' * 50}")
    print("OVERFITTING ANALYSIS OF BEST TRIALS")
    print(f"{'=' * 50}")
    
    for name, study in study_results.items():
        best_trial = study.best_trial
        overfitting_gap = best_trial.user_attrs.get('overfitting_gap', 0)
        is_overfitting = best_trial.user_attrs.get('is_overfitting', False)
        val_score = best_trial.user_attrs.get('val_score', 0)
        
        status = "⚠️ OVERFITTING" if is_overfitting else "✅ CONTROLLED"
        print(f"{name}:")
        print(f"  Best penalized score: {study.best_value:.4f}")
        print(f"  Validation score: {val_score:.4f}")
        print(f"  Overfitting gap: {overfitting_gap:.4f} {status}")

    # Create models with best parameters
    models = {
        'RandomForest': RandomForestClassifier(**rf_study.best_params, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(**xgb_study.best_params, random_state=42),
        'LightGBM': lgb.LGBMClassifier(**lgb_study.best_params, random_state=42)
    }

    # Train models
    _train_models(models, X_train, y_train, X_val, y_val)

    # Evaluate models on validation set
    _evaluate_models_on_validation(models, X_val, y_val, beta, results, study_results)

    # Select best models and create ensemble
    (ensemble, retrained_models, best_single_model_name, best_model_obj,
    ensemble_val_fbeta, ensemble_val_auc) = _select_and_ensemble_models(
        X_train, X_val, y_train, y_val, results, study_results, beta
    )

    # Final test evaluation
    (test_fbeta_best, test_auc_best, test_precision_best, test_recall_best,
    ensemble_test_fbeta, ensemble_test_auc) = _final_test_evaluation(
        best_model_obj, ensemble, X_test, y_test, beta, best_single_model_name
    )

    # Print pipeline summary
    total_time = (time.time() - start_time) / 60
    print(f"\n{'=' * 70}")
    print("OVERFITTING-AWARE PIPELINE SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total execution time: {total_time:.1f} minutes")
    print(f"Best single model: {best_single_model_name}")
    print(f"Best model test F-beta: {test_fbeta_best:.4f}")
    print(f"Ensemble test F-beta: {ensemble_test_fbeta:.4f}")
    
    # Final overfitting check
    final_overfitting = results[best_single_model_name]['overfitting']
    if final_overfitting <= overfitting_tolerance:
        print(f"✅ Overfitting successfully controlled: {final_overfitting:.4f} <= {overfitting_tolerance:.3f}")
    else:
        print(f"⚠️  Overfitting detected but minimized: {final_overfitting:.4f} > {overfitting_tolerance:.3f}")

    # Return comprehensive results
    return {
        'best_single_model': best_model_obj,
        'best_single_model_name': best_single_model_name,
        'ensemble': ensemble,
        'retrained_models': retrained_models,
        'results': results,
        'study_results': study_results,
        'test_results': {
            'best_model': {
                'fbeta': test_fbeta_best,
                'auc': test_auc_best,
                'precision': test_precision_best,
                'recall': test_recall_best
            },
            'ensemble': {
                'fbeta': ensemble_test_fbeta,
                'auc': ensemble_test_auc
            }
        },
        'execution_time_minutes': total_time,
        'overfitting_controlled': final_overfitting <= overfitting_tolerance,
        'final_overfitting_gap': final_overfitting,
        'overfitting_tolerance': overfitting_tolerance
    }


def overfitting_check(
    model: Any,
    fbeta: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Check for overfitting by comparing performance across train/val/test sets.
    
    Args:
        model: Trained machine learning model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with performance metrics for each dataset
    """
    sets = {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': (X_test, y_test)}
    results = {}
    
    for set_name, (X, y) in sets.items():
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        results[set_name] = {
            'accuracy': (y_pred == y).mean(),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': fbeta_score(y, y_pred, beta=fbeta),
            'auc': roc_auc_score(y, y_proba)
        }
    
    # Print overfitting analysis
    print(f"\\n{'=' * 50}")
    print("OVERFITTING ANALYSIS")
    print(f"{'=' * 50}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        train_score = results['train'][metric]
        val_score = results['val'][metric]
        test_score = results['test'][metric]
        
        train_val_diff = train_score - val_score
        val_test_diff = val_score - test_score
        
        print(f"{metric.upper()}:")
        print(f"  Train: {train_score:.4f}")
        print(f"  Val:   {val_score:.4f} (diff: {train_val_diff:+.4f})")
        print(f"  Test:  {test_score:.4f} (diff: {val_test_diff:+.4f})")
        
        if train_val_diff > 0.05:
            print("  ⚠️  Potential overfitting detected (train >> val)")
        elif metric in ['accuracy', 'precision', 'recall', 'f1', 'auc'] and abs(val_test_diff) > 0.05:
            # print("  ⚠️  Validation set may not be representative")
            print(f"  ⚠️  Potential overfitting detected (val >> test): {val_test_diff:+.4f}")
        else:
            print("  ✅ Good generalization")
        print()
    
    return results


def plot_feature_importance(
    results_dict: Dict,
    ensemble: VotingClassifier,
    X_train: pd.DataFrame
) -> None:
    """
    Plot feature importance for individual models and ensemble.
    
    Args:
        results_dict: Dictionary containing model results
        ensemble: Trained ensemble model
        X_train: Training features for feature names
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # Individual model importances
    model_names = ['RandomForest', 'XGBoost', 'LightGBM']
    for idx, model_name in enumerate(model_names):
        if model_name in results_dict:
            ax = axes[idx // 2, idx % 2]
            model = results_dict[model_name]['model']
            
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(15)
                
                sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
                ax.set_title(f'{model_name} Feature Importance')
                ax.set_xlabel('Importance')
    
    # Ensemble average importance
    ax = axes[1, 1]
    ensemble_importances = np.zeros(len(X_train.columns))
    
    for name, model in ensemble.named_estimators_.items():
        if hasattr(model, 'feature_importances_'):
            ensemble_importances += model.feature_importances_
    
    ensemble_importances /= len(ensemble.named_estimators_)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': ensemble_importances
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
    ax.set_title('Ensemble Average Feature Importance')
    ax.set_xlabel('Average Importance')
    
    plt.tight_layout()
    plt.show()


def explain_with_shap(model, X_sample, model_type='auto', sample_idx=0, top_n_features=10, 
                     fast_mode=True, max_evals=100):
    """
    Enhanced SHAP explainer with support for ensemble models and performance optimizations.
    
    Parameters:
    - model: Trained model (individual model or VotingClassifier ensemble)
    - X_sample: Feature matrix (100-500 samples recommended for fast mode)
    - model_type: 'auto', 'tree', 'linear', 'ensemble', 'kernel', or 'best_estimator'
    - sample_idx: Index for individual force plot explanation
    - top_n_features: Number of top features to show in dependence plots
    - fast_mode: Use optimizations for faster computation
    - max_evals: Maximum evaluations for KernelExplainer (lower = faster)
    """
    # Auto-detect model type if not specified
    if model_type == 'auto':
        from sklearn.ensemble import VotingClassifier
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        if isinstance(model, VotingClassifier):
            model_type = 'ensemble'
        elif isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier, RandomForestClassifier)):
            model_type = 'tree'
        elif isinstance(model, LogisticRegression):
            model_type = 'linear'
        else:
            model_type = 'kernel'
        
        print(f"Auto-detected model type: {model_type}")
    
    # Fast mode optimizations
    if fast_mode:
        # Limit sample size for faster computation
        if len(X_sample) > 300:
            print(f"Fast mode: Using 300 samples instead of {len(X_sample)}")
            X_sample = X_sample.sample(300, random_state=42)
        
        # For ensemble, try to use the best estimator instead
        if model_type == 'ensemble':
            try:
                print("🚀 FAST MODE: Using best individual estimator instead of full ensemble")
                
                # VotingClassifier stores estimators differently
                if hasattr(model, 'estimators_') and model.estimators_:
                    # After fitting, estimators_ contains the fitted estimators
                    estimator_names = [name for name, estimator in zip([name for name, _ in model.estimators], model.estimators_)]
                    print(f"Available estimators: {estimator_names}")
                    
                    # Use first estimator
                    best_estimator = model.estimators_[0]
                    print(f"Using: {estimator_names[0]}")
                    
                elif hasattr(model, 'estimators') and model.estimators:
                    # Before fitting, estimators contains (name, estimator) tuples
                    estimator_names = [name for name, _ in model.estimators]
                    print(f"Available estimators: {estimator_names}")
                    
                    # Use first estimator
                    best_estimator = model.estimators[0][1]
                    print(f"Using: {estimator_names[0]}")
                else:
                    raise AttributeError("No estimators found")
                
                # Recursively call with the individual estimator
                return explain_with_shap(best_estimator, X_sample, model_type='auto', 
                                       sample_idx=sample_idx, fast_mode=False)
            except Exception as e:
                print(f"⚠️  Could not extract individual estimator: {e}")
                print("Falling back to full ensemble explanation...")
                # Continue with full ensemble explanation
    
    # Initialize explainer based on model type
    if model_type == 'ensemble':
        # For ensemble models, use smaller background and limit evaluations
        background_size = 50 if fast_mode else 100
        background_sample = X_sample.sample(min(background_size, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
        explainer.max_evals = max_evals  # Limit evaluations for speed
        print(f"Using KernelExplainer with {background_size} background samples and max {max_evals} evaluations")
        
    elif model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_sample)
        
    else:  # kernel
        background_size = 50 if fast_mode else 100
        background_sample = X_sample.sample(min(background_size, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
        explainer.max_evals = max_evals
    
    # Calculate SHAP values with progress indication
    print("Computing SHAP values...")
    start_time = time.time()
    
    if model_type == 'ensemble' or model_type == 'kernel':
        # For ensemble/kernel, we need probability predictions for class 1
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (default)
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]  # Class 1 (default)
    else:
        shap_values = explainer.shap_values(X_sample)
        expected_value = explainer.expected_value
        
        # For binary classification, get SHAP for class 1 (default)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Index 1 for default class
            expected_value = expected_value[1] if isinstance(expected_value, list) else expected_value
    
    computation_time = time.time() - start_time
    print(f"✅ SHAP computation completed in {computation_time:.1f} seconds")
    
    # 1. Summary Plot (Global Feature Importance)
    print("="*60)
    print(f"GLOBAL FEATURE IMPORTANCE ({model_type.upper()} MODEL)")
    print("="*60)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()
    
    # 2. Summary Plot (Detailed Feature Effects)
    print("\n" + "="*60)
    print("FEATURE EFFECTS (Red=Higher Risk, Blue=Lower Risk)")
    print("="*60)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.show()
    
    # 3. Force Plot (Individual Prediction)
    print("\n" + "="*60)
    print(f"INDIVIDUAL PREDICTION EXPLANATION (Sample Index {sample_idx})")
    print("="*60)
    try:
        shap.force_plot(
            expected_value,
            shap_values[sample_idx, :],
            X_sample.iloc[sample_idx, :],
            matplotlib=True,
            text_rotation=15
        )
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not generate force plot: {e}")
        print("This can happen with ensemble models - skipping individual explanation.")
    
    # # 4. Dependence Plots for Top N Features
    # print("\n" + "="*60)
    # print(f"TOP {top_n_features} FEATURE DEPENDENCIES")
    # print("="*60)
    # # Get top features by mean |SHAP|
    # mean_shap = pd.DataFrame({
    #     'feature': X_sample.columns,
    #     'mean_abs_shap': np.mean(np.abs(shap_values), axis=0)
    # }).sort_values('mean_abs_shap', ascending=False)
    
    # top_features = mean_shap['feature'].values[:top_n_features]
    
    # for i, feat in enumerate(top_features):
    #     plt.figure(figsize=(8, 4))
    #     shap.dependence_plot(
    #         feat,
    #         shap_values,
    #         X_sample,
    #         interaction_index=None,  # Auto-detects strongest interaction
    #         show=False
    #     )
    #     plt.title(f"Dependence Plot: {feat}")
    #     plt.tight_layout()
    #     plt.show()


def plot_precision_recall_thresholds(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_recall: float = 0.8
) -> None:
    y_scores = model.predict_proba(X_test)[:, 1]  # Probabilities of class 1 (Default)

    # Compute precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    # Find threshold where recall is at least 80%
    min_recall = min_recall
    best_idx = np.where(recall >= min_recall)[0][-1]  # Last occurrence where recall ≥ 80%
    best_threshold = thresholds[best_idx]
    print(f"Threshold for Recall ≥ 80%: {best_threshold:.4f}")
    print(f"Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label="Precision", color="blue")  # precision[:-1] to match thresholds
    plt.plot(thresholds, recall[:-1], label="Recall", color="red")
    plt.axvline(x=best_threshold, color="black", linestyle="--", label=f"Best Threshold = {best_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs. Threshold")
    plt.legend()
    plt.grid()
    plt.show()


def plot_learning_curve(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    title: str,
    scorer: str = 'f1'
) -> None:
    """
    Plot learning curves to diagnose overfitting.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Targets  
        title: Plot title
        scorer: Scoring metric
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring=scorer,
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel(f'{scorer.upper()} Score')
    plt.title(f'Learning Curve - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_tsne(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Create t-SNE visualization of the data.
    
    Args:
        X: Feature matrix
        y: Target vector
    """
    # Sample data if too large
    if len(X) > 1000:
        sample_idx = np.random.choice(len(X), 1000, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample = X
        y_sample = y
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_sample)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


def comprehensive_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_threshold: float = 0.90
) -> pd.DataFrame:
    """
    Perform comprehensive feature selection using multiple methods.
    
    Args:
        X: Feature matrix
        y: Target vector
        correlation_threshold: Threshold for removing correlated features
        performance_tolerance: Performance tolerance for feature removal
        
    Returns:
        DataFrame with selected features
    """
    print("Starting comprehensive feature selection...")
    
    # 1. Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [column for column in upper_triangle.columns 
                if any(upper_triangle[column] > correlation_threshold)]
    
    print(f"Removing {len(to_drop)} highly correlated features")
    X_reduced = X.drop(columns=to_drop, axis=1)
    
    # 2. Statistical feature selection
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_reduced, y)
    
    feature_scores = pd.DataFrame({
        'feature': X_reduced.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    # 3. Model-based feature selection
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_reduced, y)
    
    feature_importance = pd.DataFrame({
        'feature': X_reduced.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 4. Recursive feature elimination
    rfe = RFE(estimator=rf, n_features_to_select=min(50, len(X_reduced.columns)))
    rfe.fit(X_reduced, y)
    
    selected_features = X_reduced.columns[rfe.support_].tolist()
    
    print(f"Selected {len(selected_features)} features out of {len(X_reduced.columns)}")
    
    return X_reduced[selected_features]


class DefaultPatternAnalyzer:
    """
    Clustering system specifically designed to uncover default patterns
    in loan datasets. Expects pre-processed features.
    """
    
    def __init__(self, target_col='current_loan_status', random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.scaler = None
        self.feature_columns = None
        self.best_model = None
        self.results = {}

    def analyze_defaulter_patterns(self, df, feature_columns, sampling_strategy='none'):
        """
        Specialized clustering to identify default patterns
        
        Args:
            df: DataFrame with features and target column
            feature_columns: List of column names to use for clustering
            sampling_strategy: 'none', 'oversample', or 'undersample'
        """
        print("\n=== DEFAULTER PATTERN ANALYSIS ===")
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        # Prepare data
        print(f"Using {len(feature_columns)} pre-selected features")
        df_clean = df[feature_columns + [self.target_col]].dropna()
        print(f"Clean samples: {len(df_clean)} (dropped {len(df) - len(df_clean)} rows)")
        
        # Analyze target distribution
        target_dist = df_clean[self.target_col].value_counts()
        default_rate = target_dist.get(1, 0) / len(df_clean) if len(df_clean) > 0 else 0
        print(f"Default rate: {default_rate:.1%}")
        print(f"Target distribution:\n{target_dist}")
        
        X = df_clean[feature_columns].values
        y = df_clean[self.target_col].values
        
        # Handle class imbalance if requested
        if sampling_strategy != 'none':
            X, y = self._handle_imbalance(X, y, sampling_strategy)
            print(f"After resampling: {len(X)} samples, {np.mean(y):.1%} default rate")
        
        # Scale features
        self.scaler = RobustScaler()  # Better for outliers
        X_scaled = self.scaler.fit_transform(X)
        
        # Try multiple clustering algorithms
        clustering_results = {}
        
        # 1. KMeans with different cluster numbers
        print("\nTesting KMeans clustering...")
        kmeans_results = self._test_kmeans_clustering(X_scaled, y)
        clustering_results['kmeans'] = kmeans_results
        
        # 3. DBSCAN for outlier detection
        print("\nTesting DBSCAN clustering...")
        dbscan_results = self._test_dbscan_clustering(X_scaled, y)
        clustering_results['dbscan'] = dbscan_results
        
        # 4. Hierarchical clustering
        print("\nTesting Hierarchical clustering...")
        hierarchical_results = self._test_hierarchical_clustering(X_scaled, y)
        clustering_results['hierarchical'] = hierarchical_results
        
        # Select best model
        self.best_model = self._select_best_model(clustering_results)
        self.results = clustering_results
        
        return clustering_results
    
    def _test_kmeans_clustering(self, X, y):
        """Test KMeans with different numbers of clusters"""
        results = {}
        
        for n_clusters in range(2, min(8, len(X)//10)):
            try:
                kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state, max_iter=500)
                labels = kmeans.fit_predict(X)
                
                # Evaluate clustering quality
                silhouette = silhouette_score(X, labels)
                default_separation = self._calculate_default_separation(labels, y)
                
                results[n_clusters] = {
                    'model': kmeans,
                    'labels': labels,
                    'silhouette': silhouette,
                    'default_separation': default_separation,
                    'n_clusters': len(np.unique(labels))
                }
                
                print(f"  K={n_clusters}: Silhouette={silhouette:.3f}, Default Separation={default_separation:.3f}")
                
            except Exception as e:
                print(f"  K={n_clusters}: Failed - {e}")
                
        return results
    
    def _test_gmm_clustering(self, X, y):
        """Test Gaussian Mixture Models"""
        results = {}
        
        for n_components in range(2, min(6, len(X)//20)):
            try:
                gmm = GaussianMixture(n_components=n_components, random_state=self.random_state)
                labels = gmm.fit_predict(X)
                
                silhouette = silhouette_score(X, labels)
                default_separation = self._calculate_default_separation(labels, y)
                
                results[n_components] = {
                    'model': gmm,
                    'labels': labels,
                    'silhouette': silhouette,
                    'default_separation': default_separation,
                    'n_clusters': len(np.unique(labels))
                }
                
                print(f"  Components={n_components}: Silhouette={silhouette:.3f}, Default Separation={default_separation:.3f}")
                
            except Exception as e:
                print(f"  Components={n_components}: Failed - {e}")
                
        return results
    
    def _test_dbscan_clustering(self, X, y):
        """Test DBSCAN for outlier detection"""
        results = {}
        
        # Test different eps values
        distances = []
        for i in range(min(1000, len(X))):
            distances.extend(np.sort(np.linalg.norm(X - X[i], axis=1))[1:6])  # 5 nearest neighbors
        
        eps_candidates = [np.percentile(distances, p) for p in [25, 50, 75, 90]]
        
        for eps in eps_candidates:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=max(5, len(X)//100))
                labels = dbscan.fit_predict(X)
                
                # Skip if too many or too few clusters
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2 or n_clusters > 10:
                    continue
                
                # Only calculate silhouette if not too many outliers
                outlier_ratio = np.sum(labels == -1) / len(labels)
                if outlier_ratio < 0.5:
                    silhouette = silhouette_score(X[labels != -1], labels[labels != -1]) if len(set(labels[labels != -1])) > 1 else 0
                    default_separation = self._calculate_default_separation(labels, y)
                    
                    results[eps] = {
                        'model': dbscan,
                        'labels': labels,
                        'silhouette': silhouette,
                        'default_separation': default_separation,
                        'n_clusters': n_clusters,
                        'outlier_ratio': outlier_ratio
                    }
                    
                    print(f"  eps={eps:.3f}: Clusters={n_clusters}, Outliers={outlier_ratio:.1%}, Default Sep={default_separation:.3f}")
                
            except Exception as e:
                print(f"  eps={eps:.3f}: Failed - {e}")
                
        return results
    
    def _test_hierarchical_clustering(self, X, y):
        """Test Hierarchical clustering"""
        results = {}
        
        if len(X) > 2000:  # Skip if too large
            print("  Skipping hierarchical (too many samples)")
            return results
        
        for n_clusters in range(2, min(6, len(X)//20)):
            try:
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                labels = hierarchical.fit_predict(X)
                
                silhouette = silhouette_score(X, labels)
                default_separation = self._calculate_default_separation(labels, y)
                
                results[n_clusters] = {
                    'model': hierarchical,
                    'labels': labels,
                    'silhouette': silhouette,
                    'default_separation': default_separation,
                    'n_clusters': n_clusters
                }
                
                print(f"  Clusters={n_clusters}: Silhouette={silhouette:.3f}, Default Separation={default_separation:.3f}")
                
            except Exception as e:
                print(f"  Clusters={n_clusters}: Failed - {e}")
                
        return results
    
    def _calculate_default_separation(self, labels, y):
        """
        Calculate how well clustering separates defaulters from non-defaulters
        Higher score means better separation
        """
        if len(np.unique(labels)) < 2:
            return 0
        
        cluster_default_rates = []
        for cluster in np.unique(labels):
            if cluster == -1:  # Skip outliers in DBSCAN
                continue
            mask = labels == cluster
            if np.sum(mask) > 0:
                default_rate = np.mean(y[mask])
                cluster_default_rates.append(default_rate)
        
        if len(cluster_default_rates) < 2:
            return 0
        
        # Return standard deviation of default rates (higher = better separation)
        return np.std(cluster_default_rates)
    
    def _select_best_model(self, clustering_results):
        """Select the best clustering model based on default separation"""
        best_score = 0
        best_model = None
        best_algorithm = None
        best_params = None
        
        for algorithm, results in clustering_results.items():
            for params, result in results.items():
                # Combine default separation and silhouette score
                score = result['default_separation'] * 0.7 + result['silhouette'] * 0.3
                
                if score > best_score:
                    best_score = score
                    best_model = result
                    best_algorithm = algorithm
                    best_params = params
        
        if best_model:
            print(f"\nBest model: {best_algorithm} (params: {best_params})")
            print(f"Score: {best_score:.3f} (Default Sep: {best_model['default_separation']:.3f}, Silhouette: {best_model['silhouette']:.3f})")
        
        return best_model
    
    def generate_insights(self, df):
        """Generate detailed insights about default patterns"""
        if not self.best_model:
            print("No clustering model available. Run analyze_defaulter_patterns() first.")
            return None
        
        if not self.feature_columns:
            print("No feature columns available. Run analyze_defaulter_patterns() first.")
            return None
        
        print("\n" + "="*60)
        print("DEFAULT PATTERN INSIGHTS")
        print("="*60)
        
        # Prepare data
        df_clean = df[self.feature_columns + [self.target_col]].dropna()
        labels = self.best_model['labels']
        
        # Handle case where labels might be from resampled data
        if len(labels) != len(df_clean):
            print("Warning: Label length mismatch, using original data")
            X_scaled = self.scaler.transform(df_clean[self.feature_columns])
            if hasattr(self.best_model['model'], 'predict'):
                labels = self.best_model['model'].predict(X_scaled)
            else:
                print("Cannot generate labels for this data")
                return None
        
        df_analysis = df_clean.copy()
        df_analysis['cluster'] = labels
        
        # Cluster analysis
        insights = {}
        
        for cluster in sorted(df_analysis['cluster'].unique()):
            if cluster == -1:  # Skip outliers
                continue
                
            cluster_data = df_analysis[df_analysis['cluster'] == cluster]
            cluster_size = len(cluster_data)
            default_rate = cluster_data[self.target_col].mean()
            
            print(f"\nCLUSTER {cluster} - {'HIGH RISK' if default_rate > 0.3 else 'MEDIUM RISK' if default_rate > 0.15 else 'LOW RISK'}")
            print(f"Size: {cluster_size} ({cluster_size/len(df_analysis)*100:.1f}%)")
            print(f"Default Rate: {default_rate:.1%}")
            
            # Feature characteristics
            feature_stats = {}
            for feature in self.feature_columns:
                if feature in cluster_data.columns:
                    mean_val = cluster_data[feature].mean()
                    overall_mean = df_analysis[feature].mean()
                    ratio = mean_val / overall_mean if overall_mean != 0 else 1
                    feature_stats[feature] = {
                        'cluster_mean': mean_val,
                        'overall_mean': overall_mean,
                        'ratio': ratio
                    }
                    
                    if abs(ratio - 1) > 0.2:  # Significant difference
                        direction = "higher" if ratio > 1 else "lower"
                        print(f"  {feature}: {mean_val:.3f} ({ratio:.1f}x {direction} than average)")
            
            insights[cluster] = {
                'size': cluster_size,
                'size_pct': cluster_size/len(df_analysis)*100,
                'default_rate': default_rate,
                'risk_level': 'HIGH' if default_rate > 0.3 else 'MEDIUM' if default_rate > 0.15 else 'LOW',
                'features': feature_stats
            }
        
        # Visualization
        self._visualize_clusters(df_analysis)
        
        return insights
    
    def _visualize_clusters(self, df_analysis):
        """Create comprehensive visualizations"""
        labels = df_analysis['cluster'].values
        unique_clusters = sorted([c for c in df_analysis['cluster'].unique() if c != -1])
        
        # Set up the plot
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Cluster default rates
        ax1 = plt.subplot(2, 2, 1)
        default_rates = df_analysis.groupby('cluster')[self.target_col].mean()
        bars = ax1.bar(default_rates.index, default_rates.values, 
                      color=['red' if rate > 0.3 else 'orange' if rate > 0.15 else 'green' 
                            for rate in default_rates.values], alpha=0.7)
        ax1.set_title('Default Rate by Cluster')
        ax1.set_ylabel('Default Rate')
        ax1.set_xlabel('Cluster')
        
        # Add value labels on bars
        for bar, rate in zip(bars, default_rates.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. Cluster sizes
        ax2 = plt.subplot(2, 2, 2)
        cluster_sizes = df_analysis['cluster'].value_counts().sort_index()
        ax2.pie(cluster_sizes.values, labels=[f'Cluster {c}\n({s} samples)' 
                                            for c, s in zip(cluster_sizes.index, cluster_sizes.values)], 
               autopct='%1.1f%%')
        ax2.set_title('Cluster Size Distribution')
        
        # 3. PCA visualization
        ax3 = plt.subplot(2, 2, 3)
        X_scaled = self.scaler.transform(df_analysis[self.feature_columns])
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        ax3.set_title(f'PCA Projection\n({pca.explained_variance_ratio_.sum():.1%} variance)')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar(scatter, ax=ax3)
        
        # 4. Default vs Non-default in PCA space
        ax4 = plt.subplot(2, 2, 4)
        defaulters = df_analysis[self.target_col] == 1
        ax4.scatter(X_pca[~defaulters, 0], X_pca[~defaulters, 1], 
                   c='blue', alpha=0.4, label='Non-defaulters', s=20)
        ax4.scatter(X_pca[defaulters, 0], X_pca[defaulters, 1], 
                   c='red', alpha=0.7, label='Defaulters', s=20)
        ax4.set_title('Defaulters in PCA Space')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Additional heatmap for feature correlation with clusters
        self._plot_feature_heatmap(df_analysis)
    
    def _plot_feature_heatmap(self, df_analysis):
        """Plot heatmap of feature means by cluster"""
        plt.figure(figsize=(12, 8))
        
        # Calculate feature means by cluster
        feature_means = df_analysis.groupby('cluster')[self.feature_columns].mean()
        
        # Normalize by overall mean to show relative differences
        overall_means = df_analysis[self.feature_columns].mean()
        normalized_means = feature_means.div(overall_means, axis=1)
        
        # Create heatmap
        sns.heatmap(normalized_means.T, annot=True, cmap='RdBu_r', center=1, 
                   fmt='.2f', cbar_kws={'label': 'Ratio to Overall Mean'})
        plt.title('Feature Characteristics by Cluster\n(Values relative to overall mean)')
        plt.xlabel('Cluster')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

# Main function to run the analysis
def discover_default_patterns(df, feature_columns, target_col='current_loan_status', 
                            sampling_strategy='none'):
    """
    Main function to discover default patterns in loan data
    
    Parameters:
    - df: DataFrame with loan data
    - feature_columns: List of column names to use for clustering
    - target_col: name of the target column (1 for default, 0 for non-default)
    - sampling_strategy: 'none', 'oversample', or 'undersample'
    """
    
    print("🔍 STARTING DEFAULT PATTERN DISCOVERY")
    print("="*50)
    
    # Initialize analyzer
    analyzer = DefaultPatternAnalyzer(target_col=target_col)
    
    # Run clustering analysis
    clustering_results = analyzer.analyze_defaulter_patterns(df, feature_columns, sampling_strategy)
    
    if not clustering_results or not analyzer.best_model:
        print("❌ Clustering analysis failed!")
        return None, None
    
    # Generate insights
    insights = analyzer.generate_insights(df)
    
    print("\n✅ Analysis Complete!")
    print(f"Best clustering method: {analyzer.best_model}")
    
    return analyzer, insights

def confusion_matrix_analysis(results: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    y_test_pred_best = results['best_single_model'].predict(X_test)
    ensemble_test_pred = results['ensemble'].predict(X_test)

    # Best single model confusion matrix
    cm1 = confusion_matrix(y_test, y_test_pred_best)
    sns.heatmap(cm1, annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title(f"{results['best_single_model_name']} - Test Set")
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Ensemble confusion matrix
    cm2 = confusion_matrix(y_test, ensemble_test_pred)
    sns.heatmap(cm2, annot=True, fmt='d', ax=axes[1], cmap='Blues')
    axes[1].set_title('Ensemble - Test Set')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()

def plot_permutation_importance(results_dict, ensemble, X_val, y_val, n_repeats=5, random_state=42):
    """
    Calculate and plot permutation importance for all models and ensemble.
    
    Parameters:
    - results_dict: The 'results' dictionary from your pipeline
    - ensemble: Trained VotingClassifier
    - X_val, y_val: Validation data (use validation, not test, for honest evaluation)
    - n_repeats: Number of shuffles per feature (higher=more accurate but slower)
    """
    plt.figure(figsize=(15, 10))
    models = list(results_dict.keys()) + ['Ensemble']
    
    for i, model_name in enumerate(models, 1):
        plt.subplot(2, 2, i)
        
        # Get model object
        if model_name in results_dict:
            model = results_dict[model_name]['model']
        else:
            model = ensemble
        
        # Compute permutation importance
        perm_result = permutation_importance(
            model, X_val, y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='f1'  # Or your preferred metric (fbeta_score needs custom scorer)
        )
        
        # Sort features
        sorted_idx = perm_result.importances_mean.argsort()[::-1]
        top_features = X_val.columns[sorted_idx][:15]  # Top 15 features
        top_importances = perm_result.importances_mean[sorted_idx][:15]
        
        # Plot
        plt.barh(top_features, top_importances, color='skyblue')
        plt.title(f'{model_name} - Permutation Importance (F1)')
        plt.xlabel('Mean Score Decrease')
        plt.gca().invert_yaxis()  # Highest importance on top
    
    plt.tight_layout()
    plt.show()
    
def analyze_false_negatives_misleading_features(model, X_test, y_test, top_n_features=15):
    """
    Comprehensive false negative analysis for loan default prediction.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        top_n_features: Number of most misleading features to show

    Returns:
        DataFrame with feature analysis results
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Identify groups
    fn_mask = (y_test == 1) & (y_pred == 0)  # False Negatives
    tp_mask = (y_test == 1) & (y_pred == 1)  # True Positives

    fn_data = X_test[fn_mask]
    tp_data = X_test[tp_mask]
    fn_probas = y_proba[fn_mask]

    print(f"📊 FALSE NEGATIVE ANALYSIS")
    print(f"{'=' * 50}")
    print(f"False Negatives: {fn_mask.sum()}")
    print(f"True Positives: {tp_mask.sum()}")
    print(f"FN Probability Range: {fn_probas.min():.3f} - {fn_probas.max():.3f}")
    print(f"FN Mean Probability: {fn_probas.mean():.3f}")

    # Analyze features
    feature_analysis = []

    for feature in X_test.columns:
        fn_mean = fn_data[feature].mean()
        tp_mean = tp_data[feature].mean()

        # Handle division by zero for percentage calculation
        if tp_mean != 0:
            pct_difference = ((fn_mean - tp_mean) / abs(tp_mean)) * 100
        else:
            pct_difference = 0 if fn_mean == 0 else float('inf')

        # Determine feature type
        if feature.startswith(('loan_intent_', 'home_ownership_')):
            feature_type = 'One-Hot Encoded'
        elif 'grade' in feature.lower():
            feature_type = 'Ordinal Encoded'
        else:
            feature_type = 'Numerical'

        # Calculate how misleading this feature is (absolute percentage  difference)
        misleading_score = abs(pct_difference)

        feature_analysis.append({
            'feature': feature,
            'feature_type': feature_type,
            'fn_mean': fn_mean,
            'tp_mean': tp_mean,
            'pct_difference': pct_difference,
            'misleading_score': misleading_score
        })

    # Convert to DataFrame and sort by most misleading
    analysis_df = pd.DataFrame(feature_analysis)
    analysis_df = analysis_df.sort_values('misleading_score', ascending=False)       

    # Display most misleading features
    print(f"\n🚨 TOP {top_n_features} MOST MISLEADING FEATURES:")
    print(f"{'=' * 80}")
    print(f"{'Feature':<25} {'Type':<15} {'FN Mean':<10} {'TP Mean':<10} {'%Diff':<10}")
    print(f"{'-' * 80}")

    for _, row in analysis_df.head(top_n_features).iterrows():
        print(f"{row['feature']:<25} {row['feature_type']:<15} " f"{row['fn_mean']:<10.3f} {row['tp_mean']:<10.3f}" f"{row['pct_difference']:<10.1f}%")
