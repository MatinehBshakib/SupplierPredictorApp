### Introduction

This document provides the complete instructions to run the "Automated Shipment Supplier Selection System" (ASSSS) on different operating systems. This guide covers Windows, macOS, and Linux setups.

### 1. Prerequisites

Ensure the software is installed following the instructions in install.md.

**1.1 Verifying Python and Django Installation**

python --version<br/>
pip show Django

### 2. Running the Application

**2.1 Running on Windows**

#### Activate the virtual environment:

python3.10 -m venv .venv

.\venv\Scripts\activate

#### Start the Django development server:

python manage.py runserver

Access the application at: http://localhost:8000/

**2.2 Running on macOS/Linux**

#### Activate the virtual environment:

python3.10 -m venv .venv

source .venv/bin/activate

#### Start the Django development server:

change directory to manage.py file relative path: "cd src/main/supplier_predictor"  

python manage.py runserver

Access the application at: http://localhost:8000/

