"""
model configuration
"""
import dataclasses
import enum

import immutabledict
import torch
from typing import Optional, Sequence

_STR_DTYPE_TO_TORCH_DTYPE = immutabledict.immutabledict({
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
})


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


class Architecture(enum.Enum):
    GEMMA_1 = 1
    GEMMA_2 = 2

@dataclasses.dataclass
class GenmaConfig:
    architecture: Architecture = Architecture.GEMMA_1
    vocab_size: int = 256000
    max_position_embeddings: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_size: int = 3072

    # la dimension du MLP  representations
    intermediate_size: int = 24576
    head_dim: int = 256  # num de dimension de tete
    rms_norm_eps: float = 1e-6
    # le dtype de weight (poids)
    dtype: str = 'bfloat16'

    quant: bool = False
    # path chemin du  tokenizer du  model
    tokenizer: Optional[str] = 'tokenizer/tokenizer.model'

    attn_types: Optional[Sequence[AttentionType]] = None

    # la taille de la fenetre coulissante utilisée pour l'attention locale
    sliding_window_size: Optional[int] = None

    final_logit_softcapping: Optional[float] = None  # si fourni, les logits finaux sont softcapped à cette valeur.

    attn_logit_softcapping: Optional[float] = None  # si fourni, les logits d'attention sont softcapped à cette valeur.

    query_pre_attn_scalar: Optional[
        int] = None  # si fourni, le vecteur de requête est normalisé en utilisant l'inverse au lieu de head_dim.

    use_pre_ffw_norm: bool = False  # que ce soit pour utiliser la normalisation avant le mlp

    use_post_ffw_norm: bool = False  # que ce soit pour utiliser la normalisation après le mlp

    def get_dtype(self) -> Optional[torch.dtype]:
        # Gets the torch dtype from the config dtype string
        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)


def get_config_for_7b() -> GenmaConfig:
    return GenmaConfig()


def get_config_for_2b() -> GenmaConfig:
    return GenmaConfig(
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384
    )


def get_config_for_2b_v2() -> GenmaConfig:
    return GenmaConfig(
        architecture=Architecture.GEMMA_2,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=2304,
        intermediate_size=9216,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        head_dim=256,
        attn_types=[AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * 13,
        sliding_window_size=4096,
    )


def get_config_for_9b() -> GenmaConfig:
    return GenmaConfig(
        architecture=Architecture.GEMMA_2,
        num_hidden_layers=42,
        num_attention_heads=16,
        num_key_value_heads=8,
        hidden_size=3584,
        intermediate_size=14336,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        head_dim=256,
        attn_types=[AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * 21,
        sliding_window_size=4096,
    )


def get_config_for_27b() -> GenmaConfig:
    return GenmaConfig(
        architecture=Architecture.GEMMA_2,
        num_hidden_layers=46,
        num_attention_heads=32,
        num_key_value_heads=16,
        hidden_size=4608,
        intermediate_size=36864,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        head_dim=128,
        attn_types=[AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * 23,
        sliding_window_size=4096,
        query_pre_attn_scalar=144,  # hidden_size / num_attention_heads
    )


def get_model_config(variant: str) -> GenmaConfig | ValueError:
    if variant == '7b':
        return get_config_for_7b()
    elif variant == '2b':
        return get_config_for_2b()
    return ValueError(f'Invalid Variant{variant}.Supported are "2b" and "7b" ')
