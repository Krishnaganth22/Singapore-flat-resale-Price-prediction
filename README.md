# Singapore Flat Resale Price Prediction

This project aims to build a machine learning model to predict the resale prices of flats in Singapore and deploy it as an online application using Streamlit.

## Table of Contents
1. [Overview](#overview)
2. [Technologies Used](#technologies-used)
3. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
4. [Model Building](#model-building)
5. [Streamlit Application](#streamlit-application)
6. [Usage](#usage)
7. [Files](#files)

## Overview
This project focuses on predicting the resale prices of flats in Singapore based on various features like town, flat type, block, street name, storey range, floor area, flat model, lease commencement date, year, and month. The goal is to help future buyers and sellers in evaluating the worth of a flat.

## Technologies Used
- Python
- Pandas
- Numpy
- Scikit-Learn
- Streamlit
- Matplotlib
- Machine Learning
- Data Preprocessing
- Visualization
- EDA (Exploratory Data Analysis)
- Model Deployment

## Data Cleaning and Preprocessing
The dataset used contains historical resale flat prices in Singapore. The cleaning steps involve:
1. Loading data from multiple CSV files and combining them.
2. Handling missing values.
3. Replacing inconsistent values.
4. Extracting year and month from a date column.
5. Converting columns to appropriate data types.
6. Removing non-numeric characters from the `block` column.

## Streamlit Application
The Streamlit app provides an interactive interface for users to input flat details and get the predicted resale price.

## Usage
Clone the repository.

Install the necessary dependencies using pip install -r requirements.txt.

Run the Streamlit app using streamlit run app.py.
Files
data/: Contains the datasets used for training and testing the model.

notebooks/: Jupyter notebooks for data cleaning, preprocessing, and model building.

app.py: Streamlit application code.

model.pkl: Saved model file.

label_encoder_*.pkl: Saved LabelEncoder objects for categorical features.

README.md: Project documentation.
## Contact
For any queries, please contact krishnaganth2206@gmail.com.
