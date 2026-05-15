# ModelManager Class

The `ModelManager` class is a flexible manager for training, evaluating, and saving machine learning models using **XGBoost**. It supports both **ensemble** and **hierarchical classification** modes, designed for robust and scalable learning pipelines.

---

## Supported Modes

- **Ensemble Mode**: Trains multiple models across folds and combines them for final predictions.
- **Hierarchical Mode**: First predicts a "supplier", then uses a separate model to predict an "account" based on the predicted supplier.

---

## Initialization

```python
ModelManager(mode='ensemble' | 'hierarchical', n_splits=5, random_state=42, class_names=None)
```

- `mode`: Type of model architecture.
- `n_splits`: Cross-validation splits.
- `class_names`: Optional list of human-readable class labels.

---

## Key Methods

### Training

| Method | Description |
|--------|-------------|
| `train(X, y)` | Main entry point for training (calls `_cross_validate_ensemble` or `_cross_validate_hierarchical`) |
| `_cross_validate_ensemble(X, y)` | Stratified CV training with SMOTE and XGBoost |
| `_cross_validate_hierarchical(X, y)` | Two-level training (supplier + account models) |

### Prediction

| Method | Description |
|--------|-------------|
| `predict(X, use_ensemble=True)` | Predicts class labels using ensemble or latest model |
| `_predict_hierarchical(X)` | Predicts (supplier, account) tuples for hierarchical mode |

### Evaluation

| Method | Description |
|--------|-------------|
| `evaluate(X, y)` | Evaluates the model and prints reports and confusion matrices |
| `_evaluate_hierarchical(X, y)` | Evaluates both supplier and account performance |
| `print_classification_report(y_true, y_pred)` | Wrapper for scikit-learn report |
| `plot_confusion_matrix(y_true, y_pred)` | Plots a styled confusion matrix |
| `plot_feature_importance()` | Plots top N feature importances |

### Model Persistence

| Method | Description |
|--------|-------------|
| `save_model(path)` | Saves the full model (and metadata) with `joblib` |
| `load_model(path)` | Loads a previously saved model |

---

## Internals

- **SMOTE** is used to handle imbalanced classes.
- **Sample weighting** is applied during training for fairness.
- **Feature importance** is averaged over folds for interpretability.

---

## Metrics Stored

- `accuracy`, `recall`, `f1` (per fold)
- `mean_accuracy`, `mean_recall`, `mean_f1` (overall)
- `cv_metrics['all_true']` and `cv_metrics['all_pred']` (for global analysis)

---

This class is ideal for advanced classification workflows, with flexible architecture and powerful ensemble and hierarchical learning capabilities.
