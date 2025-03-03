# REFIT U-Time: Washing Machine Power Detection

This project implements the U-Time architecture for detecting washing machine usage from household power consumption data. Using the REFIT dataset, we train a deep learning model to recognize washing machine operation patterns from aggregate electrical load measurements.

## Project Overview

U-Time is a fully convolutional neural network architecture for time series segmentation, adapted for this project to detect when washing machines are operating based on overall household power consumption data. This is a non-intrusive load monitoring (NILM) application that can help analyze energy usage patterns.

The dataset contains washing machine power consumption data extracted from the REFIT Electrical Load Measurements dataset, which includes aggregate household power consumption measurements from multiple UK households between 2013 and 2015.

## Data Setup

1. Download the UK Electrical Load dataset from [Kaggle](https://www.kaggle.com/datasets/kyleahmurphy/uk-electrical-load/data)
2. Create a `data/` directory in the project root if it doesn't already exist
3. Extract the downloaded zip file (uk-electrical-load.zip) into the `data/` directory

## Setup and Usage

This project uses `uv` for Python environment management and execution. `uv run` takes care of creating the virtual environment and installing dependencies automatically.

### 1. Process the dataset

```bash
uv run create_dataset.py
```

This script processes the original REFIT dataset to extract washing machine data and aggregate household power consumption, creating a clean dataset for model training.

### 2. Train the model

```bash
uv run train.py
```

This script:
- Loads the processed washing machine data
- Splits it into training and testing sets
- Creates a balanced dataset for washing machine detection
- Configures and trains a U-Time model
- Saves checkpoints and logs training progress

### 3. Monitor training with TensorBoard

```bash
uv run -m tensorboard --logdir=lightning_logs
```

TensorBoard will start and be accessible at http://localhost:6006 by default.

## Model Architecture

The U-Time architecture used in this project is based on the paper "U-Time: A Fully Convolutional Network for Time Series Segmentation". It consists of:

- An encoder path with dilated convolutions and max pooling
- A bottleneck layer
- A decoder path with upsampling and skip connections
- A segment classifier to produce segment-level classifications

The model is trained to detect segments where washing machine usage occurs in the aggregate power consumption data.

## Dataset Description

The dataset (`washing_machine_data.parquet`) contains:

- Datetime: Timestamp in YYYY-MM-DD HH:MM:SS format
- Aggregate: Total household power consumption in Watts
- house_number: Identifier for the household (1-21, excluding houses 4, 12, and 14)
- washing_machine: Power consumption of the washing machine in Watts

The data was collected at approximately 6-8 second intervals, with readings recorded when there was a change in load.

## Acknowledgements

This project uses data from the REFIT Electrical Load Measurements dataset:

```
@inbook{278e1df91d22494f9be2adfca2559f92,
title = "A data management platform for personalised real-time energy feedback",
keywords = "smart homes, real-time energy, smart energy meter, energy consumption",
author = "David Murray and Jing Liao and Lina Stankovic and Vladimir Stankovic and Richard Hauxwell-Baldwin and Charlie Wilson and Michael Coleman and Tom Kane and Steven Firth",
year = "2015",
booktitle = "Proceedings of the 8th International Conference on Energy Efficiency in Domestic Appliances and Lighting",
}
```

## License

This work is licensed under the Creative Commons Attribution 4.0 International Public License. See https://creativecommons.org/licenses/by/4.0/legalcode for further details.