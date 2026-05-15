
### User Requirements Specification Document
##### DIBRIS – Università di Genova. Scuola Politecnica, Software Engineering Course 80154


**VERSION: 2**

**Authors**  
Matineh Bahramishakib, 
Sheida Shahali

**REVISION HISTORY**

| Version    | Date        | Authors      | Notes        |
| ----------- | ----------- | ----------- | ----------- |
| 2 | 31/4/2025 | Matineh Bahramishakib, sheyda shahali|  |

# Table of Contents

1. [Introduction](#p1)
	1. [Document Scope](#sp1.1)
	2. [Definitios and Acronym](#sp1.2) 
	3. [References](#sp1.3)
2. [System Description](#p2)
	1. [Context and Motivation](#sp2.1)
	2. [Project Objectives](#sp2.2)
3. [Requirement](#p3)
 	1. [Stakeholders](#sp3.1)
 	2. [Functional Requirements](#sp3.2)
 	3. [Non-Functional Requirements](#sp3.3)
  
  

<a name="p1"></a>

## 1. Introduction

<a name="sp1.1"></a>
This document outlines the user requirements for the "Automated Shipment Supplier Selection System," a software designed to optimize the supplier selection process in logistics using artificial intelligence.

### 1.1 Document Scope


<a name="sp1.2"></a>
This document specifies the functional and non-functional requirements for the "Automated Shipment Supplier Selection System" (ASSSS). It is intended to provide developers with clear guidelines for the system's construction and serves as a formal agreement on project expectations between stakeholders and the development team. The document aims to ensure all parties have a common understanding of the system's capabilities and constraints. Additionally, it discusses the anticipated impact of the system on existing workflows and outlines the integration with current technological infrastructures. This scope document also identifies potential challenges and limitations in implementing the system, aiding in mitigating risks early in the development process.

### 1.2 Definitios and Acronym


| Acronym				| Definition | 
| ------------------------------------- | ----------- | 
| AI                                    | Artificial Intelligence|
| API                                   | Application Programming Interface |
| SQL                                   | Structured Query Language |
| ASSSS                                 | Automated Shipment Supplier Selection System |
| OOP                                   | Object-Oriented Programming |


<a name="sp1.3"></a>

### 1.3 References 

<a name="p2"> </a>
Company reports on logistics and supplier management <br>
Documentation on relevant APIs and SQL standards

## 2. System Description
<a name="sp2.15"></a>
This section elaborates on the functionalities of the "Automated Shipment Supplier Selection System" (ASSSS), which leverages artificial intelligence to automate and enhance the supplier selection process, specifically tailored for a company handling customer orders and requiring optimized supplier recommendations.

### 2.1 Context and Motivation

<a name="sp2.2"></a>
The logistics sector is increasingly complex with a vast array of suppliers and a multitude of variables influencing shipping products. Efficient and accurate decision-making in selecting suppliers is paramount for companies to reduce operational costs and enhance customer satisfaction. The ASSSS is designed to automate this process, selecting the most suitable supplier from a set of four based on predefined criteria, thus ensuring optimal delivery and service quality.

### 2.2 Project Obectives 

<a name="p3"></a>
The primary objective is to develop an AI-driven prototype system capable of:

Automatically selecting the best supplier for a customer's order based on specific criteria such as cost, delivery time, reliability, and emergency handling capabilities.
Sending the customer's order details directly to the selected supplier to facilitate quick and efficient fulfillment.
Integrating this selection mechanism seamlessly with the company's existing order management systems to enhance overall operational efficiency.

## 3. Requirements

| Priorità | Significato | 
| --------------- | ----------- | 
| M | **Mandatory:**   |
| D | **Desiderable:** |
| O | **Optional:**    |
| E | **future Enhancement:** |

<a name="sp3.1"></a>
### 3.1 Stakeholders

| Stackholder				| Goal | 
| ------------------------------------- | ----------- | 
| Customer                              | Company representative who uses this system |
| Supplier                              | Supply the requested product and send it to the receiver |


<a name="sp3.1"></a>
### 3.2 Functional Requirements 

| ID | Descrizione | Priorità |
| --------------- | ----------- | ---------- | 
| 0 |  Data Ingestion: The system loads the raw shipment data file (Excel or CSV) and reads the shipment data, ensuring necessary columns are present.|M|
| 1 |  Data Preprocessing: The system cleans the data by removing duplicates, correcting formats, and filling missing values.|M|
| 2 |  Geolocation (Geocoding): The system queries the Nominatim API to convert postal codes and country codes into geographical coordinates (latitude and longitude).|M|
| 3 |  Shipment Categorization: The system categorizes shipments using a machine learning model (text classification), mapping descriptions to predefined categories.|M|
| 4 |  Shipment Creation: The system creates shipment objects, calculating weight, volume, and distance for each shipment.|M|
| 5 |  Model Prediction: The system uses the trained ML model (XGBoost) to predict the best supplier and account for each shipment.|M|
| 6 |  Output Generation: The system generates a CSV file containing the predicted results and optionally displays them on the web interface (Django).|M|

<a name="sp3.3"></a>
### 3.2 Non-Functional Requirements 
 
| ID | Descrizione | Priorità |
| --------------- | ----------- | ---------- | 
| 1.0 | System Availability: The system should be operational 24/7, with a maximum allowable downtime of 0.1% annually, ensuring a system uptime of 99.9% per year to support global operations continuously. |M|
| 2.0 |  Scalability: The system should be designed to easily accommodate an increase in user numbers and data volume, supporting up to 2,000 concurrent users and handling up to 1,000,000 transactions per day without significant changes to the infrastructure. |M|
| 3.0 |  Interoperability: The system should be compatible with existing order management and supplier systems, ensuring seamless integration via standardized RESTful APIs, and achieving full compatibility with at least 95% of current systems. |E|
| 4.0 |  Data Integrity: The system must ensure that all data transmitted and received is accurate, complete, and unaltered, with mechanisms to verify data integrity at each step, achieving a data accuracy rate of 99.99% with error detection and correction mechanisms.|M|
| 5.0 |  Backup and Recovery: Effective backup and disaster recovery solutions must be in place to ensure data can be quickly restored in the event of a system failure or data loss. Data backup frequency should be every 24 hours, with a Recovery Time Objective (RTO) of 2 hours and a Recovery Point Objective (RPO) of 1 hour |O|
