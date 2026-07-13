"""HandFlow model package — attention function with flash-attn fallback."""

import torch
from einops import rearrange
from torch import Tensor

FA3_FOUND = True
try:
    from flash_attn_interface import flash_attn_func  # noqa: F401
except ImportError:
    FA3_FOUND = False

if FA3_FOUND:

    def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = rearrange(q, "B H L D -> B L H D")
        k = rearrange(k, "B H L D -> B L H D")
        v = rearrange(v, "B H L D -> B L H D")
        x = flash_attn_func(q, k, v)
        if not isinstance(x, Tensor):
            x = x[0]
        x = rearrange(x, "B L H D -> B L (H D)")
        return x

else:

    def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x
