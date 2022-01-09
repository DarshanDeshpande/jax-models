from .mpvit import MPViT_Tiny, MPViT_XSmall, MPViT_Small, MPViT_Base
from .patchconvnet import (
    PatchConvNet_S60,
    PatchConvNet_S120,
    PatchConvNet_B60,
    PatchConvNet_B120,
    PatchConvNet_L60,
    PatchConvNet_L120,
)
from .poolformer import (
    PoolFormer_S12,
    PoolFormer_S24,
    PoolFormer_S36,
    PoolFormer_M36,
    PoolFormer_M48,
)

from .mlp_mixer import (
    MLPMixer_S16,
    MLPMixer_S32,
    MLPMixer_B32,
    MLPMixer_L16,
    MLPMixer_L32,
    MLPMixer_H14,
)

model_dict = {
    "mpvit-tiny": MPViT_Tiny,
    "mpvit-xsmall": MPViT_XSmall,
    "mpvit-small": MPViT_Small,
    "mpvit-base": MPViT_Base,
    "patchconvnet-s60": PatchConvNet_S60,
    "patchconvnet-s120": PatchConvNet_S120,
    "patchconvnet-b60": PatchConvNet_B60,
    "patchconvnet-b120": PatchConvNet_B120,
    "patchconvnet-l60": PatchConvNet_L60,
    "patchconvnet-l120": PatchConvNet_L120,
    "poolformer-s12": PoolFormer_S12,
    "poolformer-s24": PoolFormer_S24,
    "poolformer-s36": PoolFormer_S36,
    "poolformer-m36": PoolFormer_M36,
    "poolformer-m48": PoolFormer_M48,
    "mlpmixer-s16": MLPMixer_S16,
    "mlpmixer-s32": MLPMixer_S32,
    "mlpmixer-b32": MLPMixer_B32,
    "mlpmixer-l16": MLPMixer_L16,
    "mlpmixer-l32": MLPMixer_L32,
    "mlpmixer-h14": MLPMixer_H14,
}


def list_models():
    return model_dict.keys()


def load_model(model_str="", attach_head=False, num_classes=1000, dropout=0.1):
    return model_dict[model_str](attach_head, num_classes, dropout=dropout)
