# Parcel Class

The `Parcel` class represents a **single parcel** within a shipment. It holds key physical attributes and supports calculation of its volume.

---

## Attributes

Each instance of `Parcel` stores:
- `weight`: The weight of the parcel
- `dim1`, `dim2`, `dim3`: The three dimensions (typically length, width, height)

All values must be **positive**. If any value is less than or equal to 0, the constructor will raise a `ValueError`.

---

## Methods

| Method | Description |
|--------|-------------|
| `__init__(weight, dim1, dim2, dim3)` | Initializes a parcel with given dimensions and weight. |
| `volume()` | Returns the **calculated volume** of the parcel: `dim1 * dim2 * dim3`. |

---

## Example Usage

```python
parcel = Parcel(weight=10, dim1=2, dim2=3, dim3=4)
print(parcel.volume())  # Output: 24
```

---

The `Parcel` class serves as the basic building block for modeling the physical aspects of shipments.
