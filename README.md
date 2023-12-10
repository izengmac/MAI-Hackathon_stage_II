# MAI-Hackathon_stage_II
Hackathon MAI
## README.md

This repository contains the code for an AerodynamicDNN model and its training script for predicting aerodynamic coefficients of objects using OpenFOAM simulation data.

### Contents

* **data_preprocessing.py:** This script parses and preprocesses OpenFOAM data into a format suitable for the DNN model.
* **model.py:** This script defines the AerodynamicDNN architecture with hidden layers and ReLU activation functions.
* **train.py:** This script trains the DNN model using the preprocessed OpenFOAM data and evaluates its performance.

### Installation

1. Install the required Python libraries:
    * `Ofpp`: This library parses OpenFOAM data files.
    * `torch`: This library provides deep learning functionalities.
    * (`numpy`, `random`): These libraries are already included in the standard Python distribution.

2. Clone this repository and navigate to its directory.

### Usage

1. Preprocess the OpenFOAM data by running:
    ```
    python data_preprocessing.py path/to/foambase/directory
    ```
    This will generate a file containing preprocessed data in a format compatible with the DNN model.

2. Train the DNN model by running:
    ```
    python train.py
    ```
    This will train the model on the preprocessed data and evaluate its performance.

### Model Architecture

The AerodynamicDNN model is a simple three-layer neural network with ReLU activation functions. The input layer accepts features extracted from the OpenFOAM data, such as velocity magnitude, pressure, and alpha.water. The hidden layers extract patterns from the data and the output layer predicts the desired aerodynamic coefficients (e.g., lift, drag, moment).

### Customization

* You can modify the `data_preprocessing.py` script to extract different features from the OpenFOAM data based on your specific needs.
* You can adjust the hyperparameters of the DNN model in the `train.py` script, such as learning rate, number of epochs, and hidden layer sizes.
* You can replace the target data parsing logic in `train.py` to accommodate your specific OpenFOAM simulation results.

### Disclaimer

This is a basic implementation of an AerodynamicDNN model for demonstration purposes. You may need to adapt and improve the code to achieve optimal performance for your specific OpenFOAM simulations and aerodynamic analysis tasks.

