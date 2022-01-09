import flax.linen as nn

from typing import Optional


class TransformerMLP(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.2
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        x = nn.Dense(self.dim)(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Dense(self.out_dim)(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        return x
