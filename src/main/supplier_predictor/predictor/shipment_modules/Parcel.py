
import logging

class Parcel:
    """
    Represents and validates an individual parcel within a shipment.

    This class ensures that parcel data is valid (e.g., positive weight,
    non-negative dimensions), computes the volume, and can be used to
    filter out invalid items before further processing.
    """

    def __init__(self, weight: float, dim1: float, dim2: float, dim3: float):
        self.weight = float(weight)
        self.dim1   = float(dim1)
        self.dim2   = float(dim2)
        self.dim3   = float(dim3)

        self.is_valid = self._validate()
        if not self.is_valid:
            logging.warning(
                "Invalid parcel detected – weight: %s, dims: (%s, %s, %s)",
                self.weight, self.dim1, self.dim2, self.dim3
            )

    # ------------------------------------------------------------------
    def _validate(self) -> bool:
        """Return True if the parcel passes business rules."""
        if self.weight <= 0:
            return False                    # weight must be positive
        if any(d < 0 for d in (self.dim1, self.dim2, self.dim3)):
            return False                    # no negative dimensions
        return True                         # zeros are allowed (documents)

    # ------------------------------------------------------------------
    def volume(self) -> float:
        """Calculate volume; 0 if any dimension is 0 (flat document)."""
        return self.dim1 * self.dim2 * self.dim3

    # Provide nice repr for debugging -----------------------------------
    def __repr__(self):
        return (
            f"Parcel(weight={self.weight}, dims=({self.dim1}, {self.dim2}, {self.dim3}), "
            f"valid={self.is_valid})"
        )
