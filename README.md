# LeaveOrNot
This is team project that we did during an event "Eyantra" conducted by IIT Bombay


## Code Explanation

Imports:
The code imports necessary libraries such as pandas, torch, sklearn, and torch-related modules for building and training a neural network.

**Salary_Predictor Class:**
Defines a neural network model for predicting salary.
It consists of fully connected layers (fc1, fc2, fc3, fc4) with ReLU activation functions and dropout layers.
Data Preprocessing Functions:

**data_preprocessing():**
Encodes categorical features and scales numerical features.
identify_features_and_targets(): Separates features and target variable from the dataset.
load_as_tensors(): Splits data into train and test sets, converts them into PyTorch tensors, and creates iterable training data using DataLoader.
Model Training Functions:

**model_loss_function():**
Defines the loss function (Binary Cross Entropy Loss).

**model_optimizer():** 
Initializes the optimizer (Adam optimizer).

**model_number_of_epochs():** 
Specifies the number of training epochs.

**Training and Validation Functions:**
training_function(): Trains the model using training data.
validation_function(): Evaluates the trained model's accuracy on the test set.
Main Section:

Reads data from a CSV file.
Performs data preprocessing.
Initializes the model, loss function, optimizer, and number of epochs.
Trains the model and evaluates its accuracy on the test set.
Saves the trained model.


## Steps to run the file
1)install all dependencies using pip(pip install) 

2)give path of csv file 

3)Run python file(python main.py)


