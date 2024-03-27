import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the input data tensor
input_data = torch.tensor([[2, 2013, 1, 3, 25, 0, 0, 3],  # Example 1
                           [1, 2017, 1, 2, 28, 0, 0, 4],  # Example 2
                           [0, 2017, 1, 2, 36, 0, 0, 3]])

# Apply normalization to numerical columns
scaler = StandardScaler()
input_data_np = input_data.numpy()
input_data_np[:, 1:5] = scaler.fit_transform(input_data_np[:, 1:5].astype(np.float32))
input_data = torch.tensor(input_data_np)

# Load the model
model = torch.jit.load('task_1a_trained_model.pth')

# Set the model in evaluation mode
model.eval()

# Perform inference
with torch.no_grad():
    for i, data in enumerate(input_data, 1):
        output = model(data.unsqueeze(0).float()) 
        print(output.item())
        if output.item() > 0.5:
            print(f"Example {i}: The model predicts that the employee will leave the company.")
        else:
            print(f"Example {i}: The model predicts that the employee will not leave the company.")
