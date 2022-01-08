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
}


def list_models():
    return model_dict.keys()


def load_model(model_str="", attach_head=False, num_classes=1000):
    return model_dict[model_str](attach_head, num_classes)
