import dataclasses
import immutabledict
import torch
from typing import Optional

_STR_DTYPE_TO_TORCH_DTYPE = immutabledict.immutabledict({
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
})


@dataclasses.dataclass
class MoeConfig:
    batch_size: int = 16
    block_size: int = 32  # max context
    max_iters: int = 5000
    eval_interval: int = 100
    learning_rate: float = 1e-3
    device = 'cuda ' if torch.cuda.is_available() else 'cpu'
    eval_iters: int = 400
    head_size: int = 16
    n_embed: int = 128
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.1
    num_experts: int = 8
    top_k: int = 2

    dtype: str = 'bfloat16'
    quant: bool = False
    tokenizer: Optional[str] = ''

    def get_dtye(self) -> Optional[torch.dtype]:
        return _STR_DTYPE_TO_TORCH_DTYPE


def get_config() -> MoeConfig:
    return MoeConfig()
