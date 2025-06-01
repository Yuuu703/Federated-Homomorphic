import flwr as fl
import torch
from .model import SecureCNN
from .privacy import DPManager
from .context import ContextManager
from .fhe_utils import FHEUtils
from src.data.data_loader import non_iid_partition
import numpy as np
import yaml
from pathlib import Path

def start_client(client_id: int, port: int):
    """Start a federated learning client"""
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    
    class FHEClient(fl.client.NumPyClient):
        def __init__(self,cid):
            self.cid = cid
            self.model = SecureCNN()
            self.dp = DPManager(config['privacy'])
            self.ctx = ContextManager().public_ctx
            self.fhe = FHEUtils(self.ctx)
            self.loader = non_iid_partition(cid, config['training']['num_clients'])
            self.client_id = client_id  # Store client_id for logging

        def evaluate(self, parameters, config):
        return 0.0, len(self.loader.dataset), {"cid": self.cid}

        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def fit(self, parameters, config):
            state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
            self.model.load_state_dict(state_dict)
            print(f"Client {self.client_id}: Starting local training...")
            self._train_epoch()
            params = [param.detach().cpu().numpy() for param in self.model.parameters()]  # Convert to NumPy
            noisy_params = self.dp.add_noise([torch.tensor(p) for p in params])
            noisy_params_np = [p.detach().cpu().numpy() for p in noisy_params]
            encrypted_params = self.fhe.encrypt_weights(self.model)
            print(f"Client {self.client_id}: Sending encrypted updates to server.")
            return noisy_params_np, len(self.loader.dataset), {"client_id": self.client_id}

        def _train_epoch(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=config['training']['lr'])
            criterion = torch.nn.CrossEntropyLoss()
            self.model.train()
            for epoch in range(config['training']['local_epochs']):
                total_loss = 0
                for batch_idx, (data, target) in enumerate(self.loader):
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    self.dp.clip_gradients(self.model)
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(self.loader)
                print(f"Client {self.client_id}: Epoch {epoch+1}/{config['training']['local_epochs']}, Avg Loss: {avg_loss:.4f}")

    fl.client.start_client(
        server_address=f"localhost:{port}",
        client=FHEClient(client_id).to_client()
    )