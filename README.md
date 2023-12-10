# MAI-Hackathon_stage_II
Hackathon MAI
## README.md


This repository provides code for training a DNN model to predict aerodynamic coefficients (lift, drag, moment) based on OpenFOAM data.

### Requirements

* Python 3.7+
* NumPy
* Torch
* openfoamparser-mai (pip install openfoamparser_mai)

### Data Preprocessing

OpenFOAM data is preprocessed using the `data_preprocessing.py` script. This script parses relevant fields like velocity, water alpha, and pressure, calculates additional features like velocity magnitude, and combines them into input features for the DNN model.

**OpenFOAM Data Path:**

* Update the `foambase_directory` variable in `data_preprocessing.py` to point to your OpenFOAM simulation base directory.
* Modify the `fields` list if you want to include additional fields for parsing.

**Target Data Parsing:**

* Replace the `...` placeholder in `data_preprocessing.py` with your specific code for parsing the target aerodynamic coefficients (lift, drag, moment) from OpenFOAM data.

### DNN Model Training

The `train.py` script trains the DNN model using the preprocessed data.

**Model Architecture:**

* The model consists of three fully connected hidden layers with ReLU activation functions and a linear output layer.
* The input layer dimension depends on the number of features included in the `data_preprocessing.py` script.
* The output layer dimension should match the number of target aerodynamic coefficients you are predicting.

**Training Configuration:**

* Modify the hyperparameters in the `train.py` script, including:
    * learning rate
    * number of epochs
    * batch size

**Running Training:**

1. Execute `python data_preprocessing.py` to preprocess the OpenFOAM data.
2. Execute `python train.py` to train the DNN model.

### Evaluation

The model performance is evaluated using Mean Absolute Error (MAE) on a separate test data set. 

**Test Data:**

* The test data is assumed to be preprocessed in the same way as the training data.

**Evaluating Model:**

* The `train.py` script automatically calculates and prints the MAE on both training and test data during each epoch.

### Contributing

Feel free to contribute to this repository by improving the code, adding features, or providing feedback. 

### Note

This code is provided for educational purposes and may require further modifications and optimization depending on your specific OpenFOAM simulation and desired accuracy.

### Updates

This repository has been updated with the following changes:

* The `data_preprocessing.py` script now uses the `openfoamparser-mai` library for improved parsing efficiency and robustness.
* The `train.py` script now calculates and prints the Mean Absolute Error (MAE) on both training and test data during each epoch.

