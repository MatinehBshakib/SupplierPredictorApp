Here’s a brief rundown of each of the four diagrams:

1. **Use-Case Diagram**
   Illustrates the high-level actors and their interactions with the system’s core functions.

   * **Actors**: Customer, Administrator, Supplier, and the internal Selection System.
   * **Use Cases** include: logging in and creating an order (by Customer), verifying orders and issuing airway bills (by Administrator), selecting a supplier/account (automated by the Selection System), and sending the shipment (to Supplier).

2. **Sequence Diagram**
   Shows the step-by-step message flow over time across the main participants.

   * **Swimlanes**: Customer, the System, Authentication System, Database, Administrator, and Supplier.
   * **Flow**: from login credential validation to order creation, order verification, AI-driven supplier selection, airway-bill issuance, and final shipment dispatch.

3. **Class Diagram**
   Maps out the key Python classes, their attributes and methods, and how they relate.

   * **Core domain classes**: `Parcel`, `Shipment`, `Location`, `Category`, `ModelManager`, and `WorkflowManager`.
   * **Django components**: `ShipmentPredictor` in the views layer, plus admin/forms/models.
   * **Relationships**: e.g. `Shipment` contains many `Parcels`; `WorkflowManager` uses `Category`, `Location`, and `ModelManager`.

4. **Activity Diagram**
   Splits the end-to-end flow into two parallel swimlanes—**Training Pipeline** and **Prediction Pipeline**—and shows decision points and tasks.

   * **Training**: loading and cleaning raw data, geocoding, category classifier training (if labeled data exists), feature generation, model training, evaluation, and persistence.
   * **Prediction**: user file upload, validation, preprocessing, geocoding, categorization (if descriptions present), feature assembly, supplier/account prediction, and result rendering/download.

