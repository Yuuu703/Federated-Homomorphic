import torch

class DPManager:
    def __init__(self, config):
        # Extract sigma and clip as floats from the config dictionary
        self.sigma = float(config.get('sigma', 0.5))  
        self.clip = float(config.get('clip', 1.0))    
        if not isinstance(self.sigma, (int, float)) or not isinstance(self.clip, (int, float)):
            raise ValueError("sigma and clip must be numeric values")

    def add_noise(self, parameters):
        return [p + torch.normal(0, self.sigma * self.clip, p.shape) for p in parameters]

    def clip_gradients(self, model):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-self.clip, self.clip)