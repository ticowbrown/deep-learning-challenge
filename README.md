# Alphabet Soup Charity Funding Predictor

This project aims to predict the success of applicants funded by Alphabet Soup using a deep learning neural network model. The goal is to identify the organizations that are most likely to be successful if funded.

## Overview of the Challenge

Alphabet Soup, a nonprofit foundation, wants a tool to help select applicants for funding with the best chance of success in their ventures. Using machine learning and neural networks, we aim to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Summary of Processing the Data

### Data Preprocessing

1. **Loading the Data:**
   - The dataset `charity_data.csv` contains information about various organizations.
   
2. **Cleaning the Data:**
   - Removed non-beneficial ID columns (`EIN` and `NAME`).
   - Binned rare categorical variables in `APPLICATION_TYPE` and `CLASSIFICATION` columns.
   - Encoded categorical variables using `pd.get_dummies()`.
   
3. **Splitting the Data:**
   - Split the dataset into features (`X`) and target (`y`) arrays.
   - Further split the data into training and testing sets.

4. **Scaling the Data:**
   - Applied `StandardScaler` to scale the feature data.

## Training the Original Model

1. **Model Architecture:**
   - Initial model with two hidden layers:
     - First hidden layer: 80 neurons, ReLU activation.
     - Second hidden layer: 30 neurons, ReLU activation.
   - Output layer with 1 neuron and Sigmoid activation for binary classification.
   
2. **Compilation:**
   - Used Adam optimizer and Binary Crossentropy loss function.
   
3. **Training:**
   - Trained the model for 100 epochs.

4. **Evaluation:**
   - Evaluated the model on the test set achieving an accuracy of approximately 72.67%.

## Creation of Optimized Model

1. **Enhanced Model Architecture:**
   - Added multiple hidden layers with increased neurons:
     - Layer 1: 200 neurons, ReLU, Dropout, Batch Normalization.
     - Layer 2: 150 neurons, ReLU, Dropout, Batch Normalization.
     - Layer 3: 100 neurons, ReLU, Dropout, Batch Normalization.
     - Layer 4: 50 neurons, ReLU, Dropout, Batch Normalization.
     - Layer 5: 20 neurons, ReLU, Dropout, Batch Normalization.
   
2. **Compilation:**
   - Used Adam optimizer with a reduced learning rate of 0.0001.
   
3. **Training:**
   - Trained the optimized model for 200 epochs with a batch size of 32 and a validation split of 20%.
   
4. **Evaluation:**
   - The optimized model achieved an accuracy of approximately 72.23% and a loss of 0.5765.

## Creation of Report

1. **Detailed Report:**
   - A comprehensive report detailing the analysis, data preprocessing steps, model architecture, training process, and evaluation results.
   - Included recommendations for future work and potential improvements.

2. **Files:**
   - `Report.md` contains the detailed report on the neural network model and its performance.
   - `AlphabetSoupCharity_Optimization.ipynb` is the Jupyter notebook with code for preprocessing, training, and evaluation.
   - Model files `AlphabetSoupCharity.h5` and `AlphabetSoupCharity_Optimization.h5` are the saved models before and after optimization.


