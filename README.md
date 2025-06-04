FHE-FL: Privacy-Preserving Federated Learning with Homomorphic Encryption

Description

This repository contains an implementation of a federated learning (FL) system enhanced with fully homomorphic encryption (FHE) to ensure privacy during distributed model training. The project trains a SecureCNN model on the MNIST dataset across 3 clients using the Flower framework, TenSEAL for FHE, and PyTorch. By encrypting model updates, it addresses privacy leaks in traditional FL, making it suitable for sensitive applications like healthcare and finance. This work is part of a Math in Computer Science course project, aiming to explore secure machine learning techniques.

Installation





Clone the Repository:

git clone https://github.com/yourusername/fhe-fl.git
cd fhe-fl



Set Up Python Environment:





Ensure Python 3.8+ is installed.



Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies:





Install required packages from requirements.txt:

python -m pip install -r requirements.txt



Ensure pyyaml, torch, tenseal, and matplotlib are included.



Verify Configuration:





Check config/config.yaml for settings (e.g., num_rounds, poly_modulus_degree).

Usage

To run the demonstration script, which showcases encryption, data distribution, privacy metrics, and optional training:

python -m src.core.demo

Expected Outputs:





data_distribution.png: Visualization of non-IID data across clients.



training_progress.png: Plots of accuracy and loss (if training runs).



final_global_model.pth: Trained model checkpoint (after training).



Console output with encryption/decryption results and privacy metrics.

Notes:





Training starts automatically if run_demo_training() is uncommented in demo.py.



Ensure the working directory is fhe-fl/ to resolve relative paths correctly.

Features





FHE Integration: Encrypts model weights using TenSEAL’s CKKS scheme.



Non-IID Data Handling: Partitions MNIST data unevenly across 3 clients.



Differential Privacy: Adds noise to updates for enhanced privacy.



Visualization: Plots data distribution and training progress.



Modular Design: Separates core logic (e.g., model.py, fhe_utils.py) for extensibility.

Contributing

This project is primarily for academic purposes. Feedback and suggestions are welcome via GitHub Issues. For contributions, please fork the repository, make changes, and submit a pull request with a clear description.
