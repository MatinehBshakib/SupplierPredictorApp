
import json
import logging
import numpy as np
from Parcel import Parcel
from geopy.distance import geodesic
from typing import List



class Shipment:
    """
    Encapsulates all data and logic related to a single shipment.

    This class is responsible for:
    1. Parsing the raw "colli" JSON data into a list of validated Parcel objects.
    2. Calculating aggregated features from its parcels, such as total weight,
       total volume, maximum dimensions, and the total number of parcels.
    3. Calculating the geodesic distance in kilometers between the shipment's
       origin and destination coordinates.
    """
    _category_model = None  # Shared class-level model

    """Represents a shipment and its attributes."""
    def __init__(self, shipment_id, supplier_account_encoded, supplier_encoded, account_encoded, goods_type_encoded,
                 service_type_encoded, shipping_type_encoded,
                 is_dangerous, dogana, dry_ice, final_category_code, parcels=None):
        self.shipment_id = shipment_id
        self.supplier_account_encoded = supplier_account_encoded
        self.supplier_encoded = supplier_encoded
        self.account_encoded = account_encoded
        self.service_type_encoded = service_type_encoded
        self.shipping_type_encoded = shipping_type_encoded
        self.goods_type_encoded = goods_type_encoded
        self.is_dangerous = is_dangerous
        self.dogana = dogana
        self.dry_ice = dry_ice
        self.origin_lat = None
        self.origin_long = None
        self.dest_lat = None
        self.dest_long = None
        self.distance = None
        self.parcels = parcels if parcels else []
        self.total_weight = 0
        self.total_volume = 0
        self.dim1_max = 0
        self.dim2_max = 0
        self.dim3_max = 0
        self.weight_max = 0
        self.number_of_parcels = 0
        self.final_category_code = final_category_code


    # Methods
        
    @staticmethod
    def clean_colli_json(x):
        """Clean and standardize Colli JSON strings."""
        if isinstance(x, str):
            try:
                x = x.replace("'", "\"")  # Replace single quotes with double quotes
                if not x.strip().startswith("["):
                    x = "[" + x + "]"  # Ensure it's a list
                json_obj = json.loads(x)  # Parse to validate JSON
                return json.dumps(json_obj)  # Return cleaned JSON as a string
            except json.JSONDecodeError:
                logging.warning(f"Invalid Colli data: {x}")
                return None
        return None  # Return None for invalid inputs

    @staticmethod
    def parse_colli(colli_data):
        """Parse Colli JSON string and return a list of *validated* Parcels.

        Args:
            colli_data: Raw JSON coming from the dataset.

        Returns:
            List[Parcel] – only parcels that meet the business rules.
        """
        parcels: List[Parcel] = []
        if not colli_data or colli_data.strip() == "":
            return parcels  # nothing to parse

        try:
            # Ensure the string is valid JSON and a list
            json_str = colli_data.replace("'", '"')
            if not json_str.strip().startswith("["):
                json_str = "[" + json_str + "]"
            colli_list = json.loads(json_str)

            for item in colli_list:
                weight = item.get("weight")
                dim1 = item.get("dim1")
                dim2 = item.get("dim2")
                dim3 = item.get("dim3")

                # Drop if weight is missing or non‑positive
                if weight is None or weight <= 0:
                    logging.warning(
                        f"Skipped parcel – invalid weight: {item}")
                    continue

                # Drop if any dimension is negative
                if any(d is not None and d < 0 for d in (dim1, dim2, dim3)):
                    logging.warning(
                        f"Skipped parcel – negative dimension: {item}")
                    continue

                # Coerce missing/None/zero dims to 0 (document case)
                dim1 = dim1 or 0
                dim2 = dim2 or 0
                dim3 = dim3 or 0

                try:
                    parcels.append(Parcel(weight, dim1, dim2, dim3))
                except Exception as e:
                    logging.error(
                        f"Could not create Parcel object for {item}: {e}")

        except json.JSONDecodeError:
            logging.warning(f"Invalid Colli JSON skipped: {colli_data}")

        return parcels

    def compute_dimensions_and_weight(self):
        """Compute total weight, volume, and max values for each dimension."""
        if not self.parcels:
            logging.warning(f"No parcels available for shipment ID {self.shipment_id}. Default values set.")
            self.total_weight = 0
            self.total_volume = 0
            self.dim1_max = 0
            self.dim2_max = 0
            self.dim3_max = 0
            self.weight_max = 0
            self.number_of_parcels = 0
            return

        self.total_weight = sum(parcel.weight for parcel in self.parcels)
        self.total_volume = sum(parcel.volume() for parcel in self.parcels)
        self.dim1_max = max(parcel.dim1 for parcel in self.parcels) if self.parcels else 0
        self.dim2_max = max(parcel.dim2 for parcel in self.parcels) if self.parcels else 0
        self.dim3_max = max(parcel.dim3 for parcel in self.parcels) if self.parcels else 0
        self.weight_max = max(parcel.weight for parcel in self.parcels) if self.parcels else 0
        self.number_of_parcels = len(self.parcels)

    def set_geographical_info(self, origin_coords, dest_coords):
        self.origin_lat, self.origin_long = origin_coords
        self.dest_lat, self.dest_long = dest_coords
        try:
            # Ensure values are numbers (convert None/strings to 0.0)
            self.origin_lat = float(origin_coords[0]) if origin_coords[0] not in [None, ""] else 0.0
            self.origin_long = float(origin_coords[1]) if origin_coords[1] not in [None, ""] else 0.0
            self.dest_lat = float(dest_coords[0]) if dest_coords[0] not in [None, ""] else 0.0
            self.dest_long = float(dest_coords[1]) if dest_coords[1] not in [None, ""] else 0.0
            
            if (np.isnan(self.origin_lat) or np.isnan(self.origin_long) or 
                np.isnan(self.dest_lat) or np.isnan(self.dest_long)):
                logging.warning(f"NaN detected for shipment {self.shipment_id}. Setting distance to 0.")
                self.distance = 0
                return

            if not (-90 <= self.origin_lat <= 90) or not (-180 <= self.origin_long <= 180):
                logging.warning(f"Invalid origin coordinates: {origin_coords} for shipment {self.shipment_id}")
                self.distance = 0
                return

            if not (-90 <= self.dest_lat <= 90) or not (-180 <= self.dest_long <= 180):
                logging.warning(f"Invalid destination coordinates: {dest_coords} for shipment {self.shipment_id}")
                self.distance = 0
                return

            # Safe to calculate distance
            self.distance = geodesic(origin_coords, dest_coords).kilometers

        except Exception as e:
            logging.error(f"Error calculating distance for shipment {self.shipment_id}: {e}")
            self.distance = 0


    def to_dict(self):
        """Convert shipment details to a dictionary."""
        return {
            "shipment_id": self.shipment_id,
            "supplier_account_encoded": self.supplier_account_encoded,
            "supplier_encoded": self.supplier_encoded,
            "account_encoded": self.account_encoded,
            "service_type_encoded": self.service_type_encoded,
            "shipping_type_encoded": self.shipping_type_encoded,
            "goods_type_encoded": self.goods_type_encoded,
            "is_dangerous": self.is_dangerous,
            "dogana": self.dogana,
            "dry_ice": self.dry_ice,
            "origin_lat": self.origin_lat,
            "origin_long": self.origin_long,
            "dest_lat": self.dest_lat,
            "dest_long": self.dest_long,
            "distance": self.distance,
            "total_weight": self.total_weight,
            "total_volume": self.total_volume,
            "dim1_max": self.dim1_max,
            "dim2_max": self.dim2_max,
            "dim3_max": self.dim3_max,
            "weight_max": self.weight_max,
            "number_of_parcels": self.number_of_parcels,
            "final_category_code": self.final_category_code
        }
