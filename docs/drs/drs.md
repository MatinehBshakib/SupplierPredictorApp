# Predictive Supplier Selection System

## Design Requirement Specification Document

DIBRIS – Università di Genova. Scuola Politecnica, Corso di Ingegneria del Software 80154

<div align='right'> <b> Authors </b> <br> Sheida Shahali <br> Matineh Bahrami Shakib </div>

### REVISION HISTORY

Version | Data | Author(s) | Notes
--------|------|-----------|------
1 | 02/05/2025 | Sheida Shahali <br> Matineh Bahrami Shakib <br> | First complete draft of the DRS


## Table of Content

1. [Introduction](#intro)  
2. [Project Description](#description)  
3. [System Overview](#system-overview)  
4. [System Module 1](#sys-module-1)  
5. [System Module 2](#sys-module-2)

---

## <a name="intro"></a> 1. Introduction

### <a name="purpose"></a> 1.1 Purpose and Scope
This document specifies the design of a machine learning-based system that predicts the best supplier for each shipment order and offers a user-friendly web interface via Django. It is intended for developers, data scientists, and stakeholders.

### <a name="def"></a> 1.2 Definitions
| Term | Definition |
|------|-----------|
| ML | Machine Learning |
| UI | User Interface |
| DRS | Design Requirements Specification |
| Django | Python web framework used for the front-end interface |
| XGBoost | A decision-tree-based ML algorithm used for model training |

### <a name="overview"></a> 1.3 Document Overview
This document outlines the project goals, architecture, data handling, ML model, and web interface design. It includes class diagrams, data flow, and dynamic behaviors.

### <a name="biblio"></a> 1.4 Bibliography
- scikit-learn, pandas, numpy documentation  
- Django framework documentation  
- SentenceTransformers and XGBoost libraries

---

## <a name="description"></a> 2. Project Description

### <a name="project-intro"></a> 2.1 Project Introduction
The system aims to automate and optimize the selection of shipping suppliers based on historical shipment data. It combines text classification, geolocation, and parcel analysis to inform the decision.

### <a name="tech"></a> 2.2 Technologies used
- Python 3.x
- Django (Web framework)
- Pandas, NumPy (data processing)
- scikit-learn, XGBoost (ML modeling)
- SentenceTransformers (text embedding)
- OpenStreetMap Nominatim API (geolocation)
- joblib (model persistence)

### <a name="constraints"></a> 2.3 Assumptions and Constraints
- Dataset availability in Excel format.
- Internet access required for geolocation.
- Execution environment must support Python 3.10+ and Django.
- Nominatim API rate limits apply.

---

## <a name="system-overview"></a> 3. System Overview

### <a name="architecture"></a> 3.1 System Architecture
The architecture includes: 
1. **Preprocessing layer** (data cleaning, encoding)  
2. **Feature engineering** (geocoding, category classification, parcel parsing)  
3. **Model layer** (XGBoost-based ensemble or hierarchical model)  
4. **Django interface** (user interaction and prediction display)

### <a name="interfaces"></a> 3.2 System Interfaces
- **Input**: Excel or CSV shipment data
- **Output**: Predicted best supplier and account, downloadable results
- **UI**: Web interface for running predictions and uploading files

### <a name="data"></a> 3.3 System Data

#### <a name="inputs"></a> 3.3.1 System Inputs
- Shipment metadata (origin, destination, services)
- Parcel details (weight, dimensions)
- Text description of goods

#### <a name="outputs"></a> 3.3.2 System Outputs
- JSON/CSV with best predicted supplier/account per shipment
- Confusion matrix and metrics visualization in the UI

---

## <a name="sys-module-1"></a> 4. System Module 1: WorkflowManager

### <a name="sd"></a> 4.1 Structural Diagrams

#### <a name="cd"></a> 4.1.1 Class Diagram
The system is structured around the following main components:

- `WorkflowManager`: Orchestrates data ingestion, preprocessing, encoding, geolocation, category classification, shipment generation, and model training.
- `ShipmentPredictor`: Handles the execution of the trained model for making predictions using new data.
- `Location`: Static geocoding utilities that cache and handle geographic coordinates based on postal codes and country.
- `ModelManager`: Manages the machine learning pipeline (training, evaluation, persistence).
- `Category`: Classifies goods descriptions using NLP models and maps to final category codes.
- `Shipment`: Represents a complete shipment and encapsulates geographical, categorical, and parcel-level information.
- `Parcel`: Validates and holds volume/dimension data per shipment unit.
- `DjangoViews (API)`: Web-facing Django views to manage file uploads and return prediction results.


#### <a name="cd-description"></a> 4.1.1.1 Class Description
| Class | Responsibility |
|-------|----------------|
| WorkflowManager | Controls end-to-end data pipeline |
| ShipmentPredictor | Loads models and performs prediction on new shipment data |
| ModelManager | Trains and evaluates machine learning models |
| Location | Performs geocoding using Nominatim and manages caches |
| Category | Classifies goods into final categories using NLP and ML |
| Shipment | Represents a shipment, including coordinates, weight, and category |
| Parcel | Validates individual package data and calculates volume |
| DjangoViews | Handles web-based user interaction for predictions |

#### <a name="od"></a> 4.1.3 Object diagram
(See attached visual representation, if applicable.)

#### <a name="dm"></a> 4.2 Dynamic Models
Sequence:
1. Data Load -> Clean -> Encode
2. Category Classifier -> Geolocation
3. Shipment Creation -> Model Training/Evaluation
4. Prediction Execution -> Results Rendering in Django UI

---

## <a name="sys-module-2"></a> 5. System Module 2: Django Interface

### <a name="sd2"></a> 5.1 Structural Design
The Django interface acts as a frontend that allows:
- Uploading shipment data files
- Triggering the prediction process
- Displaying result tables and metrics

#### <a name="cd2"></a> 5.1.1 Class Diagram
- `views.py`: Contains logic to trigger backend processing and render templates
- `forms.py`: Defines forms for file upload and parameter selection
- `urls.py`: Maps URL paths to views
- `templates/`: HTML views rendered using Django's template engine

#### <a name="cd-description2"></a> 5.1.1.1 Class Description
| File | Responsibility |
|------|----------------|
| views.py | Connects user actions with backend workflow execution |
| forms.py | Handles file upload validation |
| urls.py | Binds URLs to view functions |
| templates/ | UI layout for inputs and results |

### <a name="dm2"></a> 5.2 Dynamic Behavior
1. User uploads file via web interface.
2. Django view saves the file and invokes `WorkflowManager.run_full_workflow()`.
3. Prediction results are rendered as a downloadable CSV or shown in table format.

---
