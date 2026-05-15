# WorkflowManager Class

The `WorkflowManager` class orchestrates the **end-to-end machine learning pipeline** for shipment data processing, from raw data ingestion to geocoding, categorization, model training, evaluation, and exporting results.

---

## Purpose

Automates the complete shipment classification and modeling process using components like:
- `Location`: Geocoding
- `Category`: Shipment categorization
- `Shipment`: Parcel processing
- `ModelManager`: Model training and evaluation

---

## Initialization

```python
WorkflowManager(data_filepath, output_filepath, cleaned_dataset, encoder_path, geocoded_output, model_save_path)
```

---

## Main Workflow Steps

| Method | Description |
|--------|-------------|
| `run_full_workflow()` | Runs the full pipeline from raw data to final results |
| `load_data()` | Reads input Excel data |
| `preprocess_data()` | Cleans, deduplicates, standardizes, and handles missing values |
| `manage_rows()` | Filters rows by specific suppliers and locations |
| `encode_data()` | Encodes categorical fields using `LabelEncoder` |
| `extract_geocode_and_merge()` | Extracts geolocation and merges lat/lon via `Location` class |
| `train_category_classifier()` | Trains category model using labeled data |
| `categorize_shipments()` | Predicts categories using trained category model |
| `create_shipments()` | Converts rows into `Shipment` objects with computed stats |
| `save_shipments_to_csv()` | Exports structured shipment data |
| `train_and_evaluate_hierarchical()` | Trains and evaluates a hierarchical model |
| `train_and_evaluate_ensemble()` | Trains and evaluates an ensemble model |
| `plot_model_comparison()` | Visual comparison of model performance |
| `predict_and_export()` | Predicts supplier/account and saves output CSV |

---

## Model Training Modes

- **Hierarchical**: Classifies suppliers → accounts
- **Ensemble**: XGBoost ensemble on supplier labels
- **Both**: Trains and evaluates both models, compares results

---

## Uses These Classes

- `Category`: for classifying shipments into predefined categories
- `Location`: for extracting and merging geolocation data
- `Shipment`: for weight, volume, and dimensional computation
- `ModelManager`: for training and evaluating ML models

---

## Output Files

- `cleaned_dataset`: Cleaned and encoded dataset
- `geocoded_output.csv`: Geocoded location info
- `final_results.csv`: Shipment-level data
- `trained_model.pkl`: Persisted model
- `predicted_results.csv`: Model predictions

---

This class ties together all components of the ML workflow and enables full reproducibility, scalability, and monitoring for shipment classification and modeling projects.
