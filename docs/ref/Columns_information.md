**useless fields > fields that I think you can’t use for AI training.**

PRENOTAZIONE: our code that identify a shipment in our systems

COD_AZIENDA: identify our company ( we are 3 companies with the same db system)

Data: I have filtered all shipment elaborate after 1 January 2023

FLAG_FATTURATO: 1 if the shipment has already been invoiced, 0 otherwise

Id_quotazioni, id_quotazioni_servizio, id_quotazioni_servizio_acquisto: our codes useless for you

desc_tipo_servizio: it is a name used to identify the service used to deliver this shipment

Origin: is always PS

 

**RESULT fields > fields that the AI system should predict**

Result_Account: alphanumeric code received from our supplier. yes, the same supplier can give us more than one code. For example TNT have given us 3 accounts (8869383, 6881727, 230310)

Result_Supplier: name of the supplier

 

**Fields for training**

Service_type: identify what type of service our costumer chosen.

Shipping_type: used to identify if this shipment is an import/export/nazional

CAP_CLIFOR_MITT: used to identify in which position of the world the shipment start. It is a postal code so you can have a lot of postal code for every country. In the same country different postal code influence a lot the supplier decision

COD_NAZIONE_CLIFOR_MITT: in which nation of the world the shipment start

CAP_CLIFOR_DEST: used to identify in which position of the world the shipment end.

COD_NAZIONE_CLIFOR_DEST: in which nation of the world the shipment end

Goods_type: DOCS if the shipment contains only documents, MERCI otherwise

Is_dangerous: 1 if the shipment contains dangerous goods

Good_description: a text fields used to insert some useful good’s information

Observations: a text fields used to insert some useful information about the shipment (ex. “use FEDEX”, “do not use DHL”, “use this account” …)

Dogana: 1 if the shipment need customs operations, 0 otherwise

DryIce: 1 if the shipment need dryice, 0 otherwise

Colli: is a list of JSON Objects with weight and dimensions of every packages that create this shipment (every shipment have at list 1 package)