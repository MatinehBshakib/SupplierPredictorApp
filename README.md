# 🤖 Automated Shipment Supplier Selection System (ASSSS)

An AI-powered system developed for **B\&T Service** to automate and optimize the selection of shipping suppliers for international transport and supply chain management.

This project aims to significantly improve the efficiency and profitability of shipping operations by replacing the time-consuming manual supplier search with a data-driven model. The goal is to reduce operational workload, minimize human error, and allow employees to focus on more complex shipment cases.

This **Django-based web application** serves as a prototype to demonstrate the effectiveness of using machine learning to make optimal decisions based on historical data and defined business parameters.

-----

## 📜 Table of Contents

  - [Features](https://www.google.com/search?q=%23-features)
  - [Technology Stack](https://www.google.com/search?q=%23-technology-stack)
  - [System Architecture](https://www.google.com/search?q=%23%EF%B8%8F-system-architecture)
  - [Modeling Approach](https://www.google.com/search?q=%23-modeling-approach)
  - [Dataset Description](https://www.google.com/search?q=%23-dataset-description)
  - [Local Development Setup](https://www.google.com/search?q=%23-local-development-setup)
  - [Deployment to Heroku](https://www.google.com/search?q=%23-deployment-to-heroku)
  - [Usage](https://www.google.com/search?q=%23-usage)

-----

## ✨ Features

  - **💾 Data Ingestion & Preprocessing**: Loads and cleans shipment data from Excel or CSV files, handling duplicates, formatting, and missing values.
  - **🗺️ Geolocation**: Converts shipment origin and destination postal codes into geographical coordinates using the Nominatim API.
  - **🧩 Pluggable Categorization Module**: Designed to integrate with an external module for shipment categorization. A placeholder assigns a default category in this public version.
  - **🧠 Predictive Modeling**: Employs a trained **XGBoost** model to predict the best supplier and account for each shipment.
  - **🖥️ Web Interface**: Provides a user-friendly interface built with Django to upload data, trigger predictions, and view results.
  - **📄 Result Generation**: Outputs the prediction results into a downloadable CSV file.

-----

## 🛠️ Technology Stack

  - **Backend**: Python 3.10+, Django
  - **Data Science**: Pandas, NumPy, scikit-learn
  - **Machine Learning**: XGBoost, SentenceTransformers
  - **Geolocation**: OpenStreetMap Nominatim API, geopy
  - **Deployment**: Heroku, Gunicorn
  - **Model Persistence**: `joblib`

-----

## 🏗️ System Architecture

The application is built on a modular architecture that separates concerns, from data processing to the final user interface.

  - **Data Layer**: Ingests and cleans shipment data from Excel/CSV files using Pandas.
  - **Feature Engineering Layer**: Enriches the data using geolocation (Nominatim API) and a pluggable categorization module.
  - **Machine Learning Layer**: Predicts the optimal supplier using a trained XGBoost model managed by scikit-learn.
  - **Interface Layer**: A Django web application provides the user interface for file uploads and result display.

### Core Components

  - `ShipmentPredictor`: Orchestrates the entire prediction pipeline.
  - `ModelManager`: Manages the training, evaluation, and storage of the ML model.

-----

## 🧠 Modeling Approach

The system's predictive power comes from a **hierarchical classification model**, chosen for its high precision.

> This model splits the prediction task into two distinct stages:
>
> 1.  A **supplier classifier** predicts the general supplier (e.g., “DHL” vs. “UPS”).
> 2.  A specialized **account classifier** then predicts the specific account, conditioned on the supplier chosen in the first stage (e.g., “DHL account\_A” vs. “DHL account\_B”).

  - **Pros**: Each model tackles a narrower problem, often yielding better precision and recall.
  - **Cons**: Errors in the first stage can cascade to the second. This design requires slightly more complex orchestration.

-----

## 📊 Dataset Description

The model is trained on shipment records. The fields are categorized into input features, target variables, and ignored identifiers.

### Input Features (Fields for Training)

  - `Service_type`: The service chosen by the customer.
  - `Shipping_type`: Import, export, or national shipment.
  - `CAP_CLIFOR_MITT`: Origin postal code.
  - `COD_NAZIONE_CLIFOR_MITT`: Origin nation code.
  - `CAP_CLIFOR_DEST`: Destination postal code.
  - `COD_NAZIONE_CLIFOR_DEST`: Destination nation code.
  - `Goods_type`: `DOCS` or `MERCI` (goods).
  - `Is_dangerous`: Flag (`1` for yes, `0` for no).
  - `Good_description`: Text description of the goods.
  - `Observations`: Text field with special instructions.
  - `Dogana`: Flag for customs operations (`1` for yes, `0` for no).
  - `DryIce`: Flag for dry ice (`1` for yes, `0` for no).
  - `Colli`: A JSON list containing package weight and dimensions.

### Target Variables (Fields to Predict)

  - `Result_Account`: The specific alphanumeric account code.
  - `Result_Supplier`: The name of the predicted supplier.

### Ignored Fields

These fields are not used for training: `PRENOTAZIONE`, `COD_AZIENDA`, `FLAG_FATTURATO`, `Id_quotazioni`, `desc_tipo_servizio`, `Origin`.

-----

## ⚙️ Local Development Setup

To run this project on your local machine, follow these steps.

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd <repository-directory>
```

**2. Create and Activate a Virtual Environment**

```bash
# Create the environment
python3 -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure Encoders**
You must create a `label_encoders.json` file inside the `predictor/data/` directory. Use the sample file as a template.

```bash
cp predictor/data/label_encoders.sample.json predictor/data/label_encoders.json
```

*Now, edit `predictor/data/label_encoders.json` with your specific supplier and account mappings.*

**5. Run Database Migrations**

```bash
python manage.py migrate
```

**6. Start the Development Server**

```bash
python manage.py runserver
```

The application will now be running at `http://localhost:8000/`.

-----

## ▶️ Usage

1.  Navigate to the main URL where the application is running.
2.  Upload a CSV or Excel file containing your shipment data.
3.  Click the button to trigger the prediction process.
4.  Download the results file with the predicted supplier and account for each shipment.
