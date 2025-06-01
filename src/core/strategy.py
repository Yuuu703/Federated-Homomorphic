import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays
from .fhe_utils import FHEUtils
from .context import ContextManager
from .model import SecureCNN
import torch

class SecureFedAvg(fl.server.strategy.FedAvg):
    """FHE-enabled federated averaging strategy"""
    def __init__(self, min_fit_clients=3, min_available_clients=3, ctx_mgr=None, **kwargs):
        super().__init__(min_fit_clients=min_fit_clients, min_available_clients=min_available_clients, **kwargs)
        self.ctx_mgr = ctx_mgr or ContextManager()
        self.fhe = FHEUtils(self.ctx_mgr.private_ctx)
        self.global_model = SecureCNN()

    def aggregate_fit(self, rnd, results, failures):
        print(f"\nRound {rnd} completed with {len(results)} updates")
        if not results:
            print(f"Round {rnd}: No results to aggregate.")
            return None, {}
            
        print(f"Round {rnd}: Aggregating {len(results)} client updates.")
        weights_results = [
            (fit_res.parameters.tensors, fit_res)
            for _, fit_res in results
        ]
        aggregated_weights = super().aggregate_fit(rnd, weights_results, failures)
        
        # Update global model with aggregated weights
        if aggregated_weights[0] is not None:
            # Convert Parameters object to list of NumPy arrays
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_weights[0])
            # Create state dictionary for the global model
            state_dict = {k: torch.tensor(v) for k, v in zip(self.global_model.state_dict().keys(), aggregated_ndarrays)}
            self.global_model.load_state_dict(state_dict)
            # Convert parameters to NumPy arrays, detaching from the computation graph
            numpy_weights = [val.detach().cpu().numpy() for val in self.global_model.parameters()]
            print(f"Round {rnd}: Aggregation complete. Updated global model.")
            return numpy_weights, {}
        print(f"Round {rnd}: Aggregation failed.")
        return None, {}