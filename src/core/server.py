import flwr as fl
import yaml
from pathlib import Path
from .strategy import SecureFedAvg
import torch
from flwr.server.strategy import FedAvg

"""
def start_server(port: int):
    """Start a federated learning server"""
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    
    # Define a simple metrics aggregation function
    def metrics_aggregation_fn(fit_metrics):
        # For now, just return an empty dict since client_id isn't aggregatable
        return {}
    
    strategy = SecureFedAvg(
        min_fit_clients=config['training']['min_clients'],
        min_available_clients=config['training']['min_clients'],
        fit_metrics_aggregation_fn=metrics_aggregation_fn,
    )
    # Start the server
    hist = fl.server.start_server(
        server_address=f"localhost:{port}",
        config=fl.server.ServerConfig(num_rounds=config['training']['num_rounds']),
        strategy=strategy,
    )
    # Save the final global model
    final_weights = strategy.global_model.state_dict()
    torch.save(final_weights, "final_global_model.pth")
    print("Federated learning completed. Final global model saved to 'final_global_model.pth'.")
    return hist
"""
def start_server(port: int):
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    
    strategy = SecureFedAvg(
        min_fit_clients=config['training']['min_clients'],
        min_available_clients=config['training']['min_clients'],
        on_fit_config_fn=lambda rnd: {"round": rnd}
    )
    
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=config['training']['num_rounds']),
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024  # 1GB message limit
    )
