# hydro_conduct_layers
Python code to run a hydraulic conductivity layer model from CPT and borehole data to create inputs to MODFLOW.


This repository contains the notebooks and scripts used to generate the hydraulic conductivity layer model from CPT data used to generate inputs for MODFLOW. The contents relate to Chapter 2 of Emily Mongold's PhD thesis titled "Regional models for coastal climate risk assessment: Subsurface, multi-hazard, and risk reduction perspectives." at Stanford University. 


This repository has not been extensively cleaned, and contains additional notebooks and scripts to creating figures that appear in the thesis. The key notebooks to re-create this work are described here:

### for the regional_K package
1. In the folder 'regional_K' there is a setup.py that can be run locally. This contains 'cpt_functions' that are necessary for the entire process.

### For general results
1. clean_examples.ipynb 
    Give the distributions of K for each assigned layer for the main methodologies

### For Test Setup (inputs for MODFLOW)
1. setup_test.py
    This script sets up the .tif inputs needed for the MODFLOW workflow.
    The outputs from the MODFLOW model from Kevin Befus are in the folder 'wtdepths'

### for test postprocessing
1. test_cross_sections.ipynb
    To visualize the soil cross sections from each of the 9 test cases
2. test_outputs.ipynb
    Plots the depth to water in the test cases and compares to the modeling and empirical baselines.
3. test_postprocessing.ipynb
    This is a cleaned file that gives many of the output plots from the test cases.


### For the sensitivity analysis of layer depths
1. soil_layer_senstivity.py
    This is to set up the runs for the sensitivity study.
2. sensitivity.ipynb
    This notebook runs the PCA and Sobol index analyses on the sensitivity study.
3. sensitivity_study.ipynb 
    This notebook runs the same analyses as above, on both Fill and Young Bay Mud layers from external sensitivity runs.
4. 

### Additional figures
1. detection_figs.ipynb
    Runs the figures used in the thesis chapter to describe how the layer detection processes work


### Additional ipynbs
There is an additional 'working_files' folder that contains additional, messy working files.

Additional data from other sources:
'gtif_base' is the outputs for present-day from Befus et al. 2020
'USGS_CPT_data' are CPT data across Alameda from the USGS (2001)
'cgs_bhs_0' are borehole data from the CGS (2021)
'deposits_shp' are from the USGS (2006)
'Existing_DTW_reproj.tif' is the empirical groundwater table reprojected to EPSG:4326 from May et al. (2022)

