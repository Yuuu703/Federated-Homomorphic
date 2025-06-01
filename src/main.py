import argparse
from src.core.server import start_server
from src.core.client import start_client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FHE Federated Learning System")
    parser.add_argument("--server", action="store_true", help="Run in server mode")
    parser.add_argument("--client", type=int, help="Run as client with specified ID")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    
    args = parser.parse_args()
    
    if args.server:
        start_server(args.port)
    elif args.client is not None:
        start_client(args.client, args.port)
    else:
        print("Must specify either --server or --client <ID>")