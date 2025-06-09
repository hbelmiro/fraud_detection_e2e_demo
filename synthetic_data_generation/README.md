# Synthetic Data Generation

This module generates synthetic transaction data for fraud detection modeling and testing purposes.

## Description

The Synthetic Data Generation module creates realistic transaction data based on user profiles. It generates random transactions with various attributes such as transaction amount, category, location, and timestamps. This synthetic data can be used for training, testing, and validating fraud detection models.

## Features

- Generates random transaction data with realistic attributes
- Creates transactions with varying timestamps within a configurable time range
- Assigns transactions to user profiles
- Supports configurable number of transactions per user
- Outputs data in CSV format for easy integration with data processing pipelines

## Requirements

- Python 3.11 or compatible
- Dependencies:
  - numpy
  - pandas

## Installation

1. Use uv to create a virtual environment with Python 3.11 and install all dependencies:
   ```
   uv venv
   uv sync
   ```

   This will create a virtual environment with Python 3.11 and install all dependencies defined in the pyproject.toml file.

## Usage

1. Prepare your input data files:
   - `train.csv`: Training dataset with user profiles
   - `test.csv`: Testing dataset with user profiles
   - `validate.csv`: Validation dataset with user profiles

2. Run the synthetic data generation script:
   ```
   python synthetic_data_generation.py
   ```

3. The script will generate a file named `raw_transaction_datasource.csv` containing the synthetic transaction data.

## Input Files

- `train.csv`: Contains user profiles for the training set
- `test.csv`: Contains user profiles for the testing set
- `validate.csv`: Contains user profiles for the validation set

These files should contain at minimum a column named `repeat_retailer` which is used to filter users for transaction generation.

## Output Files

- `raw_transaction_datasource.csv`: Contains the generated synthetic transaction data with the following columns:
  - `user_id`: Identifier for the user
  - `created`: Timestamp when the record was created
  - `updated`: Timestamp when the record was last updated
  - `date_of_transaction`: Date and time when the transaction occurred
  - `transaction_amount`: Amount of the transaction (between 10 and 1000)
  - `transaction_category`: Category of the transaction (e.g., Groceries, Utilities, Entertainment)
  - `card_token`: Unique identifier for the payment card
  - `city`: City where the transaction occurred
  - `state`: State where the transaction occurred

## Customization

You can customize the transaction generation by modifying the parameters in the script:

- `max_transactions`: Maximum number of transactions to generate per user (default: 5)
- `max_days_back`: Maximum number of days in the past for transaction dates (default: 365)

## Integration

This module is part of a larger fraud detection end-to-end demo project. The generated data can be used as input for feature engineering and model training in the fraud detection pipeline.
