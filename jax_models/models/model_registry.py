from . import conv_mixer
from . import mlp_mixer
from . import mpvit
from . import patchconvnet
from . import poolformer


model_dict = {
    "mpvit-tiny": mpvit.MPViT_Tiny,
    "mpvit-xsmall": mpvit.MPViT_XSmall,
    "mpvit-small": mpvit.MPViT_Small,
    "mpvit-base": mpvit.MPViT_Base,
    "patchconvnet-s60": patchconvnet.PatchConvNet_S60,
    "patchconvnet-s120": patchconvnet.PatchConvNet_S120,
    "patchconvnet-b60": patchconvnet.PatchConvNet_B60,
    "patchconvnet-b120": patchconvnet.PatchConvNet_B120,
    "patchconvnet-l60": patchconvnet.PatchConvNet_L60,
    "patchconvnet-l120": patchconvnet.PatchConvNet_L120,
    "poolformer-s12": poolformer.PoolFormer_S12,
    "poolformer-s24": poolformer.PoolFormer_S24,
    "poolformer-s36": poolformer.PoolFormer_S36,
    "poolformer-m36": poolformer.PoolFormer_M36,
    "poolformer-m48": poolformer.PoolFormer_M48,
    "mlpmixer-s16": mlp_mixer.MLPMixer_S16,
    "mlpmixer-s32": mlp_mixer.MLPMixer_S32,
    "mlpmixer-b32": mlp_mixer.MLPMixer_B32,
    "mlpmixer-l16": mlp_mixer.MLPMixer_L16,
    "mlpmixer-l32": mlp_mixer.MLPMixer_L32,
    "mlpmixer-h14": mlp_mixer.MLPMixer_H14,
    "convmixer-1536-20": conv_mixer.ConvMixer_1536_20,
    "convmixer-1024-20": conv_mixer.ConvMixer_1024_20,
    "convmixer-768-32": conv_mixer.ConvMixer_768_32,
    "convmixer-512-12": conv_mixer.ConvMixer_512_12,
}


def list_models():
    return sorted(list(model_dict.keys()))


def load_model(model_str="", attach_head=False, num_classes=1000, dropout=0.1):
    return model_dict[model_str](attach_head, num_classes, dropout=dropout)
