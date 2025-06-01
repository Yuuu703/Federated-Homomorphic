from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def non_iid_partition(client_id: int, num_clients: int):
    """Create non-IID data partition for client"""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Create non-IID distribution
    labels = np.asarray(dataset.targets)
    label_sorted = np.argsort(labels)
    partitions = np.array_split(label_sorted, num_clients * 2)
    
    # Assign 2 random partitions per client
    np.random.shuffle(partitions)
    client_indices = np.concatenate(partitions[client_id*2:(client_id+1)*2])
    
    return DataLoader(
        Subset(dataset, client_indices),
        batch_size=128,
        shuffle=True
    )