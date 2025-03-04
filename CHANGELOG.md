## Changelog


### 0.1.45 - 2025-03-05

* Added write_potcar()
* Added add_molecule_to_surface()
* Added fill_multiple_spots option to add_interstitial()

### 0.1.44 - 2024-12-16

* Added the get_ferwe_ferdo() function to oganesson.io.vasp
* For DGL, only import it if it is already installed.

### 0.1.43 - 2024-09-30

* Fix for Xdatcar reader
* Misc fixes

### 0.1.42 - 2024-08-08

* New features in io.vasp
* Added the phase_stability module

### 0.1.41 - 2024-04-30

* From imp to importlib because imp is deprecated in python 3.12

### 0.1.40 - 2024-04-08

* Fix for relax() in OgStructure

### 0.1.39 - 2024-04-08

* Added sampling of frozen atoms to the fracture() method
* Added "opt_pulling_expansion" approach for the fracture() method

### 0.1.38 - 2024-03-28

* Added the fracture() method
* Support for relaxation and dynamics using constraints
* Fix for AIMD simulation

### 0.1.37 - 2024-03-28

* Fix for AIMD simulation

### 0.1.36 - 2024-03-28

* SoftMutation removed from GA because of its limited support for elements

### 0.1.35 - 2024-03-27

* Hot fix for loading local PES models

### 0.1.34 - 2024-03-27

* Hot fix for loading local PES models

### 0.1.33 - 2024-03-26

* Added the DIEP PES model for relaxation, and set as default

### 0.1.32 - 2024-03-26

* Hot fix

### 0.1.31 - 2024-03-26

* Replace M3GNET Relaxer with Matgl Relaxer
* Added VASP postprocessing functions

### 0.1.30 - 2024-03-26

* Enabled different GNN models based on m3gnet for structure relaxation
* Critical fix for GA

### 0.1.29 - 2024-03-21

* Added theoretical capacity formula

### 0.1.28 - 2024-03-14

* Critical fixes

### 0.1.27 - 2024-02-22

* create_ripple now computes the amplitude

### 0.1.26 - 2024-02-16

* Various fixes
* Added the get_graph() method to obtain the structure's graph using m3gnet implementation
* Added get_bonds_blocks()
* Fix for SOAP parameters in DScribe
* improved the tutorial notebook

### 0.1.25 - 2023-11-30

* Adding interstitial atoms to a structure
* Several minor fixes

### 0.1.24 - 2023-07-27

* Hot fix for the delta vector function

### 0.1.23 - 2023-07-27

* Addition of delta vector function

### 0.1.22 - 2023-07-16

* Addition of adsorption_scanner() method
* Addition of transformed fragment descriptors
* Addition of the structure rippling code

### 0.1.21 - 2023-07-14

* Enabled more logging in GA

### 0.1.20 - 2023-07-12

* Hot fix: included get_rdf within OgStructure because it is missing in the most recent version of ASE. Also would benefit from some optimization.

### 0.1.9 - 2023-07-12

* Over-ridden the particle mutation of ASE
* Few fixes

### 0.1.8 - 2023-07-07

* Added new GA operators
* Critical fix for vasp.py
* Several fixes

### 0.1.7 - 2023-06-13

* Creating the GA object won't relax the initial population. The first call for evolve() will do
* Initial GA structures can be written to CIF files

### 0.1.6 - 2023-06-13

* GA population can be provided by user
* Added random atom substitution to OgStructure
* Added XRD to OgStructure via pymatgen's API

### 0.1.5 - 2023-06-08

* Added MD to OgStructure
* Added more examples

### 0.1.4 - 2023-05-18

* Added DScribe descriptors
* Avoid importing gpaw/dscribe if they were not installed properly

### 0.1.3 - 2023-05-17

* Fixing setup.py

### 0.1.2 - 2023-05-17

* Fixing setup.py
* GA: Enabling writing structure into a uuid-based folder

### 0.1.1 - 2023-05-16

* Addition of the OgStructure class
* Addition of the ML descriptors BACD, ROSA and SymmetryFunctions

### 0.1.0 - 2023-05-09

* First release
