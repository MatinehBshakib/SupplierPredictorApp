### Introduction

This document provides the necessary instructions to compile the "Automated Shipment Supplier Selection System" (ASSSS) by ensuring all dependencies and project settings are properly configured.

### Supported Architectures

Windows 10/11 (64-bit)

macOS (Intel and Apple Silicon)

Linux (Ubuntu, Debian, CentOS)

### 1. Compilation Steps

**1.1 Preparing the Environment**

Ensure Python 3.10+ is installed:

python --version
pip --version

**Set up a virtual environment:**

python3.10 -m venv .venv

Activate the virtual environment:

#### On Windows:

.\venv\Scripts\activate

#### On macOS/Linux:

source .venv/bin/activate

**1.2 Installing Dependencies**

Install required Python packages

**1.3 Preparing Database (No Execution)**

Only prepare the database structure without running the server:

python manage.py makemigrations </br>
python manage.py migrate

**1.4 Compilation Verification**

The system is compiled when all dependencies are installed, the database is prepared, and static files are collected (if needed).


