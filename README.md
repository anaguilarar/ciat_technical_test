# ciat technical test

This README provides guidance on how to use the Python script designed for obtaining mean decreasing impurity score using random forest models. The script employs cross-validation apporach that are trained with a provided dataset.

## Requirements

* Python 3.6 or higher

Make sure to install the required libraries using pip:
```bash
pip install -r requirements.txt
```

## Configuration

The script requires a YAML configuration file to specify parameters for training, including dataset paths, model details, and evaluation metrics. This repository has an example of the structure, the file is called 'configuration.yaml'

## Running the Script

Use the following command to run the script:

```bash
python calculate_mdi.py --config configuration.yaml

```

## R scripts

Additioanlly to the python scripts, there are two files located in r_scripts, that describes the code used for generating plots and data explorotary analysis.
