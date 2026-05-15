## Category Class

The `Category` class is responsible for assigning a **final shipment category** to each order, based on the textual content of its `goods_description` field. It combines **semantic text analysis** with **machine learning** and **keyword rules** to map each order into one of a set of **fixed, predefined categories**.

---

## Inputs & Outputs

- **Input:** `pandas.DataFrame` with a column named `goods_description`
- **Output:** Same `DataFrame` with an added column `final_category_code` (numerical encoding of the category)

---

## What It Does

1. **Text Cleaning & Preprocessing:**  
   Uses lemmatization, stopword removal (EN & IT), and punctuation stripping to normalize the text.

2. **Category Prediction (ML-based):**  
   Trained on 2,000 labeled samples using a **SentenceTransformer (SBERT)** and a **Random Forest classifier** to predict one of the **fixed main categories**.

3. **Subcategory Detection (Rule-based):**  
   For broad categories like `"Spare_Parts"` or `"Mechanical_Parts"`, further refinement into **predefined subcategories** is done using keyword matching.

4. **Final Category Selection:**  
   Each order is ultimately assigned **one final category**, prioritized by subcategory if available, else by the main category.

---

##  Fixed Categories

Some of the predefined categories include:
- `Engine_parts`
- `Ship_spare_parts`
- `Heavy_equipment_parts`
- `Appliances`
- `Automotive_parts`
- `Tools_and_Instruments`
- `Fragile_and_Delicate_items`
- `Lost_and_Found`
- `Spare_Parts`
- `Electronics_and_Gadgets`
- `Food_and_Beverages`
- `Hazardous_Materials`
- `Clothing_and_Accessories`
- `Unknown`
- `General_Mechanical_Parts`

> These categories are **not dynamically created** — they are fixed and well-defined.

---

##  Key Functions

| Method                             | Description |
|----------------------------------|-------------|
| `__init__()`                     | Initializes the transformer model and loads keyword mappings |
| `_process_text_fields()`         | Cleans and normalizes `goods_description` for ML use |
| `_preprocess_and_assign_unknown()` | Fills blanks and assigns 'Unknown' where needed |
| `train_classifier()`             | Trains classifier on manually labeled 2K sample dataset |
| `classify_mechanical_subcategory()` | Rule-based keyword check for assigning subcategories |
| `categorize_data()`             | Main method: predicts, assigns subcategory, and maps to final category code |

---

##  Output Behavior

The final result is a single encoded label in `final_category_code` that maps each order into one of the **predefined fixed categories**, no ambiguity involved.
