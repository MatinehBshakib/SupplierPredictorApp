
import pandas as pd
import requests
import time
import logging
from typing import Dict, Tuple, Set


class Location:
    """
    A utility for handling geographical location data for shipments.

    This class provides static methods to:
    1. Extract unique (postal_code, country) pairs from a DataFrame.
    2. Geocode these pairs into latitude and longitude using the Nominatim API.
    3. Manage a local cache to minimize API calls, apply manual fixes for
       problematic addresses, and log any unresolved locations.
    4. Merge the resulting coordinates back into the original DataFrame.
    """

    # ---------------- Configuration ----------------
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    USER_AGENT = "sh.m@gmail.com"
    DELAY = 1  # seconds

    CACHE_FILE = "geocoded_output.csv"
    UNRESOLVED_FILE = "unresolved.csv"

    # ---------------- Internal state --------------
    _cache: Dict[Tuple[str, str], Tuple[float | None, float | None]] = {}
    _unresolved: Set[Tuple[str, str]] = set()

    # ---------------- CSV helpers -----------------
    @staticmethod
    def _save_csv(df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)
        logging.info("File saved → %s (%d rows)", path, len(df))

    # ---------------- Cache -----------------------
    @classmethod
    def load_cache(cls, csv_path: str | None = None) -> None:
        csv_path = csv_path or cls.CACHE_FILE
        try:
            df = pd.read_csv(csv_path)
            cls._cache = {
                (str(r["postal_code"]), str(r["country"])): (r["latitude"], r["longitude"])
                for _, r in df.iterrows()
            }
            logging.info("Loaded %d cached combos", len(cls._cache))
        except FileNotFoundError:
            logging.info("No cache file at %s", csv_path)
            cls._cache = {}

    @classmethod
    def dump_cache(cls, csv_path: str | None = None) -> None:
        if not cls._cache:
            logging.info("Cache empty – nothing to dump")
            return
        csv_path = csv_path or cls.CACHE_FILE
        df = pd.DataFrame([
            {"postal_code": pc, "country": c, "latitude": lat, "longitude": lon}
            for (pc, c), (lat, lon) in cls._cache.items()
        ])
        cls._save_csv(df, csv_path)

    # ---------------- Manual fixes ----------------
    @classmethod
    def load_manual_fixes(cls, csv_path: str) -> None:
        """Merge user‑supplied lat/lon into _cache and remove them from
        the unresolved set so they won’t re‑appear in *unresolved.csv*."""
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            logging.info("manual_fixes.csv not found → skipping")
            return

        req = {"postal_code", "country", "latitude", "longitude"}
        if df.empty or not req.issubset(df.columns):
            logging.warning("manual_fixes.csv missing required columns")
            return

        added = 0
        for _, r in df.iterrows():
            key = (str(r["postal_code"]).strip(), str(r["country"]).strip())
            try:
                lat = float(r["latitude"])
                lon = float(r["longitude"])
            except (TypeError, ValueError):
                continue  # skip bad row
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue  # invalid numbers
            cls._cache[key] = (lat, lon)
            cls._unresolved.discard(key)  # clean
            added += 1
        logging.info("Manual fixes applied: %d", added)

    # ---------------- Unresolved queue ------------
    @classmethod
    def _mark_unresolved(cls, key: Tuple[str, str]):
        cls._unresolved.add(key)

    @classmethod
    def dump_unresolved(cls, csv_path: str | None = None) -> None:
        """Persist unresolved keys to *unresolved.csv*; remove if all resolved."""
        csv_path = csv_path or cls.UNRESOLVED_FILE

        def is_missing(latlon):
            if latlon is None:
                return True
            lat, lon = latlon
            return (
                lat is None or lon is None or
                pd.isna(lat) or pd.isna(lon) or
                (lat == 0 and lon == 0)
            )

        cls._unresolved = {k for k in cls._unresolved if is_missing(cls._cache.get(k))}

        import os
        if not cls._unresolved:
            if os.path.exists(csv_path):
                os.remove(csv_path)
                logging.info("All combos resolved – deleted %s", csv_path)
            else:
                logging.info("All combos resolved – no unresolved file needed")
            return

        df = pd.DataFrame([{"postal_code": pc, "country": c} for pc, c in sorted(cls._unresolved)])
        df.to_csv(csv_path, index=False)
        logging.info("Unresolved combos dumped → %s (%d rows)", csv_path, len(df))

    # ---------------- Combo extraction ------------
    @classmethod
    def extract_unique_combos(cls, df: pd.DataFrame) -> Set[Tuple[str, str]]:
        """Collect unique (postal_code, country) pairs from mitt + dest columns.
        This restores the public API expected by WorkflowManager."""
        combos: Set[Tuple[str, str]] = set()
        for _, row in df.iterrows():
            if pd.notna(row["CAP_CLIFOR_MITT"]) and pd.notna(row["COD_NAZIONE_CLIFOR_MITT"]):
                combos.add((str(row["CAP_CLIFOR_MITT"]), str(row["COD_NAZIONE_CLIFOR_MITT"])))
            if pd.notna(row["CAP_CLIFOR_DEST"]) and pd.notna(row["COD_NAZIONE_CLIFOR_DEST"]):
                combos.add((str(row["CAP_CLIFOR_DEST"]), str(row["COD_NAZIONE_CLIFOR_DEST"])))
        return combos

    # ---------------- Geocoding -------------------
    @classmethod
    def _call_nominatim(cls, postal_code: str, country: str) -> Tuple[float | None, float | None]:
        try:
            params = {"format": "json", "country": country}
            if postal_code:
                params["postalcode"] = postal_code
            resp = requests.get(cls.NOMINATIM_URL, params=params, headers={"User-Agent": cls.USER_AGENT}, timeout=10)
            time.sleep(cls.DELAY)
            if resp.status_code == 200 and resp.json():
                lat = float(resp.json()[0]["lat"])
                lon = float(resp.json()[0]["lon"])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
            logging.warning("Nominatim failure (%s) for %s/%s", resp.status_code, postal_code or "-", country)
        except Exception as e:
            logging.error("Nominatim exception: %s", e)
        return None, None

    @classmethod
    def geocode_all(
        cls,
        combos: Set[Tuple[str, str]],
        *,
        cache_csv: str | None = None,
        manual_fixes_csv: str | None = None,
        update_cache: bool = True,
        unresolved_csv: str | None = None,
    ) -> Dict[Tuple[str, str], Tuple[float | None, float | None]]:
        if cache_csv:
            cls.load_cache(cache_csv)
        if manual_fixes_csv:
            cls.load_manual_fixes(manual_fixes_csv)

        results: Dict[Tuple[str, str], Tuple[float | None, float | None]] = {}

        for pc, country in combos:
            key = (pc, country)
            if key in cls._cache:
                results[key] = cls._cache[key]
                if results[key] == (None, None):
                    cls._mark_unresolved(key)
                continue
            lat_lon = cls._call_nominatim(pc, country)
            if lat_lon == (None, None) and pc:
                lat_lon = cls._call_nominatim("", country)
            cls._cache[key] = lat_lon
            results[key] = lat_lon
            if lat_lon == (None, None):
                cls._mark_unresolved(key)
                
        cls._unresolved = {
                    k for k in cls._unresolved
                    if cls._cache.get(k, (None, None)) == (None, None)
                }
        if update_cache and cache_csv:
            cls.dump_cache(cache_csv)
        if unresolved_csv or cls.UNRESOLVED_FILE:
            cls.dump_unresolved(unresolved_csv or cls.UNRESOLVED_FILE)
        return results

    # ---------------- Merge helper ----------------
    @staticmethod
    def _clean_float(v: float | None) -> float:
        try:
            return 0.0 if v is None or pd.isna(v) else float(v)
        except Exception:
            return 0.0

    @classmethod
    def merge_lat_lon(cls, df: pd.DataFrame, geocoded: Dict[Tuple[str, str], Tuple[float | None, float | None]]):
        def fetch(row, pc_col, c_col, idx):
            return cls._clean_float(geocoded.get((str(row[pc_col]), str(row[c_col])), (None, None))[idx])
        df["origin_lat"] = df.apply(lambda r: fetch(r, "CAP_CLIFOR_MITT", "COD_NAZIONE_CLIFOR_MITT", 0), axis=1)
        df["origin_long"] = df.apply(lambda r: fetch(r, "CAP_CLIFOR_MITT", "COD_NAZIONE_CLIFOR_MITT", 1), axis=1)
        df["dest_lat"] = df.apply(lambda r: fetch(r, "CAP_CLIFOR_DEST", "COD_NAZIONE_CLIFOR_DEST", 0), axis=1)
        df["dest_long"] = df.apply(lambda r: fetch(r, "CAP_CLIFOR_DEST", "COD_NAZIONE_CLIFOR_DEST", 1), axis=1)
        return df
