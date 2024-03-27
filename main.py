import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

class Salary_Predictor(nn.Module):
    def __init__(self, input_size):
        super(Salary_Predictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

def data_preprocessing(task_1a_dataframe):
    df = task_1a_dataframe.copy()
    
    label_encoders = {}
    categorical_columns = ['Education', 'City', 'Gender', 'EverBenched']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  
    
    scaler = StandardScaler()
    numerical_columns = ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
   
    return df

def identify_features_and_targets(encoded_dataframe):
    features = encoded_dataframe.drop(columns=['LeaveOrNot'])
    target = encoded_dataframe['LeaveOrNot']
    return [features, target]

def load_as_tensors(features_and_targets):
    X_train, X_test, y_train, y_test = train_test_split(features_and_targets[0], 
                                                        features_and_targets[1], 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    iterable_training_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    return [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, iterable_training_data]

def model_loss_function():
    return nn.BCELoss()

def model_optimizer(model):
    return Adam(model.parameters(), lr=0.001)

def model_number_of_epochs():
    return 100  

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    for epoch in range(number_of_epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(tensors_and_iterable_training_data[4]):
            optimizer.zero_grad()
            model.train()  # Switch to training mode
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    return model

def validation_function(trained_model, tensors_and_iterable_training_data):
    trained_model.eval()
    correct = 0
    with torch.no_grad():
        y_pred = trained_model(tensors_and_iterable_training_data[1])
        predicted = (y_pred > 0.5).float()
        correct += (predicted == tensors_and_iterable_training_data[3]).float().sum().item()
    accuracy = correct / len(tensors_and_iterable_training_data[3])
    return accuracy

if __name__ == "__main__":

    task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

    model = Salary_Predictor(features_and_targets[0].shape[1])

    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    for epoch in range(number_of_epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(tensors_and_iterable_training_data[4]):
            optimizer.zero_grad()
            model.train()  # Switch to training mode
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()  # Switch to evaluation mode
    model_accuracy = validation_function(model, tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    # Save the trained model
    torch.jit.save(torch.jit.trace(model.eval(), (tensors_and_iterable_training_data[0][0],)), "task_1a_trained_model.pth")
