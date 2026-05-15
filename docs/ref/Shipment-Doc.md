# Shipment Class

The `Shipment` class models a shipment entity and manages its key metadata, parcels, and metrics such as weight, volume, and travel distance.

---

## Inputs & Outputs

- **Inputs (via `__init__`)**:
  - `shipment_id`, `supplier_account_encoded`, `supplier_encoded`, `account_encoded`, etc.
  - `parcels`: Optional list of Parcel objects.
- **Outputs**:
  - Derived attributes like `total_weight`, `total_volume`, `distance`, and dimensional stats.
  - Structured output via `to_dict()` method.

---

## What It Does

1. **Handles Parcel Data**:  
   - Cleans and parses raw Colli JSON to create `Parcel` objects.
   - Computes weight, volume, and dimension statistics from parcels.

2. **Geographical Distance**:  
   - Accepts coordinates and uses the `geopy` library to calculate shipment distance.

3. **Serialization**:  
   - Outputs all shipment attributes in a dictionary format.

---

## Key Methods

| Method                     | Description |
|----------------------------|-------------|
| `__init__()`              | Initializes shipment attributes and optional parcels |
| `clean_colli_json(x)`     | Cleans raw JSON string of parcel data |
| `parse_colli(data)`       | Parses parcel JSON and creates `Parcel` instances |
| `compute_dimensions_and_weight()` | Aggregates dimensional and weight-related stats |
| `set_geographical_info(origin, dest)` | Sets origin/destination coordinates and calculates distance |
| `to_dict()`               | Converts shipment object to a serializable dictionary |

---

## Dimensions and Stats

Once the parcel data is parsed and processed:
- `total_weight`: Total weight across parcels
- `total_volume`: Sum of volumes of all parcels
- `dim1_max`, `dim2_max`, `dim3_max`: Max values for each dimension
- `weight_max`: Heaviest parcel weight
- `number_of_parcels`: Count of parcel objects

---

## Distance Calculation

Uses:
```python
geopy.distance.geodesic(origin_coords, dest_coords).kilometers
```

Handles missing, invalid, or malformed coordinates gracefully by logging warnings and setting distance to `0`.

---

## Output Structure (`to_dict()`)

Returns a dictionary including:
- Shipment metadata (IDs, codes, flags)
- Geo-coordinates
- Shipment stats (weight, volume, distance, dimensions)

---

This class plays a foundational role in modeling and preprocessing shipment records for logistics, optimization, and analysis tasks.
