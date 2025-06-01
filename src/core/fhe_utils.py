import tenseal as ts
import torch
import numpy as np
from typing import List, Tuple

class FHEUtils:
    """Handle FHE operations with batch processing"""
    def __init__(self, context):
        self.ctx = context
        
    def encrypt_weights(self, model: torch.nn.Module) -> List[Tuple[bytes, tuple]]:
        encrypted = []
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            arr = param.cpu().numpy().flatten()
            enc = ts.ckks_vector(self.ctx, arr)
            encrypted.append((enc.serialize(), param.shape))
        return encrypted
        
    def decrypt_weights(self, encrypted: List[Tuple[bytes, tuple]]) -> List[torch.Tensor]:
        decrypted = []
        for enc_data, shape in encrypted:
            vec = ts.ckks_vector_from(self.ctx, enc_data)
            decrypted.append(torch.tensor(vec.decrypt()).reshape(shape))
        return decrypted
        
    def aggregate(self, ciphertexts: List[List[Tuple[bytes, tuple]]]) -> List[Tuple[bytes, tuple]]:
        aggregated = []
        for i in range(len(ciphertexts[0])):
            vectors = [ts.ckks_vector_from(self.ctx, client[i][0]) for client in ciphertexts]
            avg = sum(vectors) * (1/len(vectors))
            aggregated.append((avg.serialize(), ciphertexts[0][i][1]))
        return aggregated