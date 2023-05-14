import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from collections import OrderedDict

from handwriting_recognition import HandwritingRecognitionModel

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# - all on the same chip
MPS_DEVICE = "mps"
CPU_DEVICE = "cpu"

# - checkpoint paths
GLOBAL_MODEL_CHECKPOINT = "global_model_checkpoint.pt"
MPS_MODEL_CHECKPOINT = "mps_model_checkpoint.pt"
CPU_MODEL_CHECKPOINT = "cpu_model_checkpoint.pt"

checkpoints = {GLOBAL_MODEL_CHECKPOINT: MPS_DEVICE,
           MPS_MODEL_CHECKPOINT: MPS_DEVICE,
           CPU_MODEL_CHECKPOINT: CPU_DEVICE,}

def run_test(model, device, test_loader, epoch, name=""):
    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model on the validation set
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # Pass the input through the model
            outputs = model(images)

            # Get the predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Update the total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum()

        # Print the accuracy
        print(f"{device} {(name)}: Epoch {epoch+1}: Accuracy = {100*correct/total:.2f}%")
        return 100.0*correct/total

    
# A run_round returns the local gradient for a model, after performing epochs 
# training/test and creates a model checkpoint
def run_round(model_checkpoint, device, train_loader, test_loader, epochs=1):
    # Initialize a model to the device
    model = HandwritingRecognitionModel().to(device)
    
    # Set the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    base_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    last_epoch = base_epoch+epochs

    for epoch in range(base_epoch, last_epoch):
        # Set the model to training mode
        model.train()

        # Iterate over the training data
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Pass the input through the model
            outputs = model(images)

            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Backpropagate the error
            loss.backward()

            # Update the model parameters
            optimizer.step()

        accuracy = run_test(model, device, test_loader, epoch, name="Worker")

    # Save the model checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'device': device,
    }

    torch.save(checkpoint, model_checkpoint)

    return [model.state_dict(), accuracy]

def initialize_checkpoints(global_model, global_optimizer):
    for checkpoint_path, device in checkpoints.items():
        model_to_device = global_model
        optimizer_to_device = global_optimizer
        
        # if global_model.device != device:
        #     model_to_device = model_to_device.to(device)

        # if optimizer_to_device != device:
        #     optimizer_to_device = optimizer_to_device(device)

        checkpoint = {
            'epoch': 0,
            'model_state_dict': model_to_device.state_dict(),
            'optimizer_state_dict': optimizer_to_device.state_dict(),
            'loss': None,
            'device': device,
        }

        torch.save(checkpoint, checkpoint_path)
            
def compute_global_model_state(worker_1_state, worker_1_accuracy, worker_2_state, worker_2_accuracy, device):    
    global_model_state = OrderedDict()

    for name, param in worker_1_state.items():
        param = param.to(device)
        global_model_state[name] = param*worker_1_accuracy

    for name, param in worker_2_state.items():
        param = param.to(device)
        global_model_state[name] += param*worker_2_accuracy

    for name, param in global_model_state.items():
        global_model_state[name] /= (worker_1_accuracy+worker_2_accuracy)

    return global_model_state

def update_local_checkpoint(checkpoint_path, device, global_model_state_dict):
    local_checkpoint = torch.load(checkpoint_path)

    for layer, tensor in global_model_state_dict.items():
        tensor_on_device = tensor.to(device)
        local_checkpoint['model_state_dict'][layer] = tensor_on_device

    torch.save(local_checkpoint, checkpoint_path)

# Load the MNIST dataset
train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
test_dataset = MNIST("./data", train=False, download=True, transform=ToTensor())

num_samples = len(train_dataset)
split_ratio = 0.5
split_index = int(split_ratio*num_samples)

# Create the indices for train_mps and train_cpu subsets
torch.manual_seed(42)
indices = torch.randperm(num_samples)

train_mps_indices = indices[:split_index]
train_cpu_indices = indices[split_index:]

# Create tehe mps and cpu subsets using the Subset class
train_mps_subset = Subset(train_dataset, train_mps_indices)
train_cpu_subset = Subset(train_dataset, train_cpu_indices)

# Define the data loaders
batch_size = 64
train_loader_mps = DataLoader(train_mps_subset, batch_size)
train_loader_cpu = DataLoader(train_cpu_subset, batch_size)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# Init global model and optimizer
global_model = HandwritingRecognitionModel().to(MPS_DEVICE)
global_optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)

# Init checkpoints
initialize_checkpoints(global_model, global_optimizer)

# Train the model for 10 rounds
epochs = 1
device_1 = MPS_DEVICE
device_2 = CPU_DEVICE
for round in range(20):
    print(f'Running round {round}')
    mps_model_state, mps_accuracy = run_round(MPS_MODEL_CHECKPOINT, device_1, 
                                train_loader_mps, test_loader, epochs)
    cpu_model_state, cpu_accuracy = run_round(CPU_MODEL_CHECKPOINT, device_2,
                                train_loader_cpu, test_loader, epochs)
    
    # Compute global parameters
    global_model_state = compute_global_model_state(mps_model_state, mps_accuracy, 
                                                cpu_model_state, cpu_accuracy, MPS_DEVICE)

    # Load global checkpoint
    global_checkpoint = torch.load(GLOBAL_MODEL_CHECKPOINT)
    global_model.load_state_dict(global_model_state)
    global_optimizer.load_state_dict(global_checkpoint['optimizer_state_dict'])

    # Test the model
    run_test(global_model, MPS_DEVICE, test_loader, -1, name="Global")

    # Save the global checkpoint
    global_checkpoint_updated = {
        'epoch': round,
        'model_state_dict': global_model.state_dict(),
        'optimizer_state_dict': global_optimizer.state_dict(),
        'loss': None,
        'device': MPS_DEVICE,
    }

    torch.save(global_checkpoint_updated, GLOBAL_MODEL_CHECKPOINT)
    print("------------------------------------------------------------\n")