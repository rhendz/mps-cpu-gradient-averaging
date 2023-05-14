import torch
from torch import nn

# Define the CNN model
class HandwritingRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Define the pooling and dropout layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Define the fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout2(x)

        # Reshape the output for the fully connected layers
        x = x.view(-1, 32 * 7 * 7)

        # Pass the output through the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Return the final output
        return x
    
model = HandwritingRecognitionModel()
print(sum(p.numel() for p in model.parameters()))