# Location Class

The `Location` class handles all **geocoding operations** — converting postal codes and country codes into geographic coordinates (latitude and longitude) using the **Nominatim API**.

---

## Responsibilities

- Extracts unique combinations of postal codes and countries from a dataset.
- Queries the **OpenStreetMap Nominatim** service for geolocation data.
- Merges retrieved lat/lon data back into the main dataset.
- Saves cleaned and enriched data as CSV files.

---

## Geocoding Process Overview

1. **Extract Unique Location Keys**  
   Finds unique `(postal_code, country)` pairs from origin and destination fields.

2. **Query Nominatim API**  
   Uses postal code and country to get geo-coordinates. If postal code fails, it retries with country alone.

3. **Merge Coordinates**  
   Joins latitude/longitude back to the original DataFrame and stores results.

---

## Key Methods

| Method | Description |
|--------|-------------|
| `extract_unique_geocode_combos(df)` | Extracts unique origin/destination postal+country combos from a DataFrame. |
| `process_geocoding(unique_combos)` | Queries Nominatim API and returns a dictionary of geocoded results. |
| `_call_nominatim(postal_code, country)` | Low-level call to the API; validates and returns `(lat, lon)`. |
| `merge_lat_lon(df, geocoded_data)` | Adds coordinates to the main DataFrame and saves the result. |
| `save_csv(df, output_path)` | Utility method to save any DataFrame to CSV. |

---

## Input Fields Used

- `CAP_CLIFOR_MITT`, `COD_NAZIONE_CLIFOR_MITT` → Origin ZIP and country
- `CAP_CLIFOR_DEST`, `COD_NAZIONE_CLIFOR_DEST` → Destination ZIP and country

---

## Retry Logic

If the full `(postal code + country)` lookup fails, it tries again using only the country to improve hit rate.

---

## Output Files

1. `unique_combos_output.csv` — All unique postal-country pairs
2. `geocoded_output.csv` — Retrieved latitude and longitude for each combo
3. `merged_output.csv` — Final DataFrame with lat/lon merged into original rows

---

This class plays a vital role in enriching shipment data with geographical coordinates, enabling downstream tasks like distance calculations and routing logic.
