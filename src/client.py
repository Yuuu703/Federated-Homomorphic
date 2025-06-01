# src/client.py
from core.client import start_client
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=int, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    start_client(args.client, args.port)