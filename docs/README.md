## Detailed Subdirectory Descriptions

### **1. DRS (Design Requirements Specification)**
- Contains the `drs.md` file, which specifies the design and architecture of the system.
- The `img` subdirectory stores diagrams and visual representations of the architecture, class structure, and workflow.
- Key contents in `drs.md` include:
  - System architecture (components, modules, and interactions).
  - Structural and dynamic models (class diagrams, sequence diagrams).
  - Design principles and technologies used.

### **2. Manuals**
- Contains user and installation manuals:
  - `compile.md`: Instructions on how to compile the system (if applicable).
  - `install.md`: Detailed setup instructions, including dependencies, environment setup, and configuration.
  - `run.md`: Instructions on how to run the system (Django server, command-line commands, etc.).

### **3. Ref (Reference Documentation)**
- Provides detailed descriptions of all key components and classes in the system:
  - `Category-Doc.md`: Describes the Category class (NLP and classification).
  - `Location-Doc.md`: Explains the Location class (geocoding).
  - `ModelManager-Doc.md`: Details the ModelManager class (ML model training and evaluation).
  - `Parcel-Doc.md`: Describes the Parcel class (parcel attributes, volume calculation).
  - `Shipment-Doc.md`: Documents the Shipment class (shipment data management).
  - `Workflow-Doc.md`: Explains the WorkflowManager class (data processing pipeline).
  - `SupplierList.md` and `SupplierList.png`: List and visualization of suppliers.
  - `Columns_information.md`: Explanation of each data column used in the system.
  - `Interview.md`: Transcript of the initial interview for requirement gathering.
  - `Questions_list.md`: Initial questions to clarify system goals and requirements.

### **4. URS (User Requirements Specification)**
- Contains the `urs.md` file, which specifies user requirements, stakeholders, functional and non-functional requirements.
- Key contents in `urs.md` include:
  - Context and motivation for the system.
  - Project objectives.
  - Stakeholder list and their roles.
  - Detailed functional and non-functional requirements.

## How to Use This Directory
- Use the **URS** (`urs.md`) to understand user needs and expectations.
- Refer to the **DRS** (`drs.md`) for technical design and system architecture.
- Consult the **Manuals** for setup, usage, and troubleshooting.
- Use the **Ref** directory for detailed technical understanding of each class/module.
