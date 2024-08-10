# ANAC Open Data to Event Log

![PyPI - Python Version](https://img.shields.io/badge/python-3.12-3776AB?logo=python)

### > Directories

#### config
Configuration directory with ```config.yml```.  

#### open_data_anac
Open Data to be copied in this directory for this experiment are available here: [https://bit.ly/4dkNMeQ](https://bit.ly/4dkNMeQ).  

#### stats
Directory with procurements stats.

#### utility_manager
Directory with utilities functions.

### > Script Execution

#### ```01_data_filter.py```
Loads the various datasets (in CSV format) and generates the event log. Only keeps cases starting with the TENDER_NOTICE event.  

#### ```02_data_filter.py```
Loads the filtered dataset in the previous script and trains Machine Learning (ML) models.  

#### ```03_data_plot.ipynb```
Loads the ML results ad plot the results.  

#### ```04_data_map.ipynb```
Loads the the various datasets (in CSV format) plot values by Italian region on a map (use ```limits_IT_regions.geojson```).  

### > Configurations

#### ```conf_cols_filter.json```
List of columns (features) to be filtered.  

#### ```conf_cols_stats.json```
List of columns (features) to be filtered.  

#### ```conf_cols_type.json```
List of columns (features) types.  

#### ```limits_IT_regions.geojson```
GeoJSON file of Italian regions (used by ```04_data_map.ipynb```).  

### > Script Dependencies
See ```requirements.txt``` for the required libraries (```pip install -r requirements.txt```).  
