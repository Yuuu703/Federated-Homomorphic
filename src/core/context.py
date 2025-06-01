import tenseal as ts
import yaml
from pathlib import Path

# Robust config path
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"

class ContextManager:
    def __init__(self):
        self.config = yaml.safe_load(CONFIG_PATH.read_text())
        self.context = None

    @property
    def private_ctx(self):
        if self.context is None:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.config['fhe']['poly_modulus_degree'],
                coeff_mod_bit_sizes=self.config['fhe']['coeff_mod_bit_sizes']
            )
            self.context.global_scale = self.config['fhe']['global_scale']
            self.context.generate_galois_keys()
        return self.context

    @property
    def public_ctx(self):
        ctx = self.private_ctx
        ctx.make_context_public()
        return ctx