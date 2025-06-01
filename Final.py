import torch
import numpy as np
import matplotlib.pyplot as plt
from src.core.model import SecureCNN

try:
    # Load with error handling
    model = SecureCNN()
    state_dict = torch.load("C:/Math-20250527T121651Z-1-001/fhe-fl/final_global_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Print all weights with summarization
    for name, param in model.named_parameters():
        weights = param.data.numpy()
        print(f"\nLayer: {name}")
        print(f"Shape: {param.shape}")
        print(f"Mean: {np.mean(weights):.4f}, Std: {np.std(weights):.4f}")
        print(f"Sample (first 5 values flattened): {weights.flatten()[:5]}")

    # Visualize conv1 weights
    if 'conv1.weight' in state_dict:
        weights = model.conv1.weight.data.numpy()
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            if i < weights.shape[0]:
                ax.imshow(weights[i, 0], cmap='gray')
                ax.set_title(f'Filter {i}')
                ax.axis('off')
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"Error: {e}")