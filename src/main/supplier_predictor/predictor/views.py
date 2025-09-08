from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

import pandas as pd
import json
import os
import sys
import logging

from .forms import ShipmentUploadForm
from .models import PredictionResult

# Add the path to custom shipment processing modules.
sys.path.append(os.path.join(settings.BASE_DIR, 'predictor', 'shipment_modules'))

from  shipment_modules.ModelManager import ModelManager
from  shipment_modules.Location import Location
from  shipment_modules.Shipment import Shipment

class ShipmentPredictor:
    """
    Orchestrates the end-to-end prediction workflow within the Django app.

    At runtime, this class is responsible for:
    1. Loading the pre-trained XGBoost model and label encoders.
    2. Preprocessing the data from a user-uploaded file.
    3. Calling the core modules to perform geocoding, feature engineering,
       and categorization.
    4. Executing the hierarchical prediction to get the final supplier/account.
    5. Returning the results to be displayed or downloaded.
    """
    
    def __init__(
        self,
        encoder_path,
        model_path,
        cache_csv,
        manual_fixes_csv='manual_fixes.csv'
    ):
        """
        Initializes the ShipmentPredictor with paths to required assets.

        Args:
            encoder_path (str): Path to the JSON file containing label encoders.
            model_path (str): Path to the trained model file (.pkl).
            cache_csv (str): Path to the CSV file for geocode caching.
            manual_fixes_csv (str, optional): Path to manual geocode fixes.
        """
        self.encoder_path = encoder_path
        self.model_path = model_path
        self.cache_csv = cache_csv
        self.manual_fixes_csv = manual_fixes_csv

        # Initialize the geocoding cache to speed up location lookups.
        Location.load_cache(csv_path=self.cache_csv)
        Location.load_manual_fixes(csv_path=self.manual_fixes_csv)

        # Load the pre-trained prediction model.
        try:
            self.model_manager = ModelManager.load_model(self.model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'. Please verify the path.")
        if self.model_manager.mode != 'hierarchical':
            logging.warning("Loaded model is not in hierarchical mode, but predictions will use it.")

    def preprocess_data_for_prediction(self, input_df):
        """
        Cleans and prepares the input DataFrame for prediction.

        This involves removing unnecessary columns, handling duplicates, ensuring
        correct data types, and applying label encoding.
        """
        logging.info("Preprocessing data for prediction...")
        df = input_df.copy()
        
        # Drop target columns if they exist and remove duplicate entries.
        df.drop(columns=['Result_Supplier', 'Result_Account'], errors='ignore', inplace=True)
        df.drop_duplicates(subset=['PRENOTAZIONE'], keep='first', inplace=True)
        
        # Standardize data types and fill missing values.
        df['dogana'] = pd.to_numeric(df['dogana'], errors='coerce').fillna(0).astype(int)
        df['DryIce'] = pd.to_numeric(df['DryIce'], errors='coerce').fillna(0).astype(int)
        df['PRENOTAZIONE'] = df['PRENOTAZIONE'].astype(str)
        
        for col in df.columns:
            if df[col].dtype in ["object", "string"] or df[col].isna().any():
                df[col] = df[col].astype(str).fillna("UNKNOWN")
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
        
        for col in ['goods_type', 'service_type', 'shipping_type', 'goods_description']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Apply label encoding using pre-saved encoder mappings.
        with open(self.encoder_path, 'r') as f:
            encoders = json.load(f)
        
        # Build lookup dictionaries to handle unseen values gracefully.
        goods_map   = {v: i for i, v in enumerate(encoders['goods_type_encoded'])}
        serv_map    = {v: i for i, v in enumerate(encoders['service_type_encoded'])}
        ship_map    = {v: i for i, v in enumerate(encoders['shipping_type_encoded'])}
        
        df['goods_type_encoded']    = df['goods_type'].apply(lambda x: goods_map.get(x, goods_map.get('UNKNOWN', 0)))
        df['service_type_encoded']  = df['service_type'].apply(lambda x: serv_map.get(x, serv_map.get('UNKNOWN', 0)))
        df['shipping_type_encoded'] = df['shipping_type'].apply(lambda x: ship_map.get(x, ship_map.get('UNKNOWN', 0)))
        return df

    def geocode_and_merge(self, df):
        """
        Converts postal codes to geographic coordinates and merges them back.
        """
        combos = Location.extract_unique_combos(df)
        geocoded = Location.geocode_all(
            combos,
            cache_csv=self.cache_csv,
            manual_fixes_csv=self.manual_fixes_csv,
            update_cache=True,
            unresolved_csv='unresolved.csv'
        )
        return Location.merge_lat_lon(df, geocoded)

    def categorize(self, df):
        """
        Placeholder for an external shipment categorization module.
        
        This method assigns a default category code to all shipments. It can be
        replaced with a custom classification model.
        """
        logging.info("Assigning default category (placeholder)...")
        df['final_category_code'] = 0  # Assign a default numeric code (e.g., 0).
        return df

    def add_shipment_features(self, df):
        """
        Generates engineered features like distance, weight, and dimensions.
        """
        logging.info("Generating shipment features: distance, dimensions, weight...")
        records = []
        for _, row in df.iterrows():
            parcels = Shipment.parse_colli(Shipment.clean_colli_json(row.get('colli', '')))
            shipment = Shipment(
                shipment_id=row['PRENOTAZIONE'],
                parcels=parcels,
                **row
            )
            shipment.set_geographical_info(
                (row.get('origin_lat'), row.get('origin_long')),
                (row.get('dest_lat'), row.get('dest_long'))
            )
            shipment.compute_dimensions_and_weight()
            records.append(shipment.to_dict())
        
        # Extract only the newly generated features.
        feature_cols = [
            'distance', 'total_weight', 'total_volume',
            'dim1_max', 'dim2_max', 'dim3_max', 'weight_max', 'number_of_parcels'
        ]
        feats = pd.DataFrame(records)[feature_cols]
        
        return df.reset_index(drop=True).join(feats)
    
    def _ensure_required_features(self, df):
        """
        Ensures the DataFrame has all features the model was trained on.
        
        If a feature is missing, it's added with a default value of 0.0.
        """
        for col in self.model_manager.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        return df
    
    def predict(self, df):
        """
        Runs the hierarchical prediction model on the feature-engineered data.
        """
        logging.info("Predicting supplier and account in hierarchical mode...")
        
        # Impute any remaining NaNs in numeric columns before prediction.
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        
        # Ensure the feature set matches the model's training set.
        feature_cols = self.model_manager.feature_names
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        
        X = df[feature_cols].values
        preds = self.model_manager.predict(X)
        
        # Unpack hierarchical predictions into separate columns.
        if isinstance(preds, list) and preds and isinstance(preds[0], tuple):
            df[['supplier_pred', 'account_pred']] = pd.DataFrame(preds, index=df.index)
        else:
            # Provide a fallback for single-label model outputs.
            df['supplier_account_pred'] = preds
        return df

    def _decode_supplier_predictions(self, df):
        """
        Converts numeric prediction labels back to their original string names.
        """
        with open(self.encoder_path, 'r') as f:
            encoders = json.load(f)
        sup_classes = encoders.get('supplier_encoded', [])
        acc_classes = encoders.get('account_encoded', [])

        # Map numeric codes back to class names safely.
        df['supplier_pred'] = df['supplier_pred'].apply(
            lambda c: sup_classes[int(c)] if isinstance(c, (int, float)) and 0 <= int(c) < len(sup_classes) else c
        )
        df['account_pred'] = df['account_pred'].apply(
            lambda c: acc_classes[int(c)] if isinstance(c, (int, float)) and 0 <= int(c) < len(acc_classes) else c
        )
        return df

    def run_prediction(self, file_path):
        """
        Executes the full prediction pipeline on a given input file.

        Args:
            file_path (str): The path to the input CSV or Excel file.

        Returns:
            pd.DataFrame: A DataFrame with the prediction results.
        """
        # Load the input data from the specified file.
        if file_path.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
            
        # Execute each step of the pipeline in sequence.
        df_prep = self.preprocess_data_for_prediction(df)
        df_geo = self.geocode_and_merge(df_prep)
        df_cat = self.categorize(df_geo)
        df_feat = self.add_shipment_features(df_cat)
        df_feat = self._ensure_required_features(df_feat)
        df_pred = self.predict(df_feat)
        df_pred = self._decode_supplier_predictions(df_pred)
        
        return df_pred[['PRENOTAZIONE', 'supplier_pred', 'account_pred']]

# --- Django Views ---

# Initialize a single, global instance of the predictor.
predictor = ShipmentPredictor(
    encoder_path=os.path.join(settings.BASE_DIR, 'predictor', 'data', 'label_encoders.json'),
    model_path=os.path.join(settings.BASE_DIR, 'predictor', 'data', 'trained_model.pkl'),
    cache_csv=os.path.join(settings.BASE_DIR, 'predictor', 'data', 'geocode_cache.csv')
)

def index(request):
    """
    Handles the main page, which contains the file upload form.
    """
    if request.method == 'POST':
        form = ShipmentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            
            # Save the file to a temporary location for processing.
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'tmp'))
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)
            
            # Run the prediction pipeline.
            try:
                results_df = predictor.run_prediction(file_path)
                
                # Store results in the user's session to display on the next page.
                request.session['prediction_results'] = results_df.to_dict('records')
                
            except Exception as e:
                # Display an error if processing fails.
                return render(request, 'predictor/index.html', {
                    'form': form,
                    'error': f"Error processing file: {str(e)}"
                })
            finally:
                # Clean up the temporary file regardless of success or failure.
                if os.path.exists(file_path):
                    os.remove(file_path)
                
            return redirect('results')
    else:
        form = ShipmentUploadForm()
    
    return render(request, 'predictor/index.html', {'form': form})

def results(request):
    """
    Displays the prediction results stored in the user's session.
    """
    results = request.session.get('prediction_results', [])
    return render(request, 'predictor/results.html', {'results': results})

def about(request):
    """
    Renders the 'About' page.
    """
    return render(request, 'predictor/about.html')

def download_results(request):
    """
    Serves the prediction results as a downloadable CSV file.
    """
    results = request.session.get('prediction_results', [])
    if not results:
        return redirect('index')
    
    df = pd.DataFrame(results)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction_results.csv"'
    df.to_csv(path_or_buf=response, index=False)
    return response
