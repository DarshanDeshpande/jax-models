from .conv_mixer import *
from .mlp_mixer import *
from .mpvit import *
from .patchconvnet import *
from .poolformer import *
from .segformer import *
from .convnext import *
from .masked_autoencoder import *
from .swin_transformer import *


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
    "convmixer-1536-20": ConvMixer_1536_20,
    "convmixer-1024-20": ConvMixer_1024_20,
    "convmixer-768-32": ConvMixer_768_32,
    "convmixer-512-12": ConvMixer_512_12,
    "segformer-b0": SegFormer_B0,
    "segformer-b1": SegFormer_B1,
    "segformer-b2": SegFormer_B2,
    "segformer-b3": SegFormer_B3,
    "segformer-b4": SegFormer_B4,
    "segformer-b5": SegFormer_B5,
    "convnext-tiny": ConvNeXt_Tiny,
    "convnext-small": ConvNeXt_Small,
    "convnext-base": ConvNeXt_Base,
    "convnext-large": ConvNeXt_Large,
    "convnext-xlarge": ConvNeXt_XLarge,
    "mae-base": MAE_Base,
    "mae-large": MAE_Large,
    "mae-huge": MAE_Huge,
    "swin-tiny-224": SwinTiny224,
    "swin-small-224": SwinSmall224,
    "swin-base-224": SwinBase224,
    "swin-base-384": SwinBase384,
    "swin-large-224": SwinLarge224,
    "swin-large-384": SwinLarge384,
}


def list_models():
    """
    Lists all available model architectures.
    """
    return sorted(list(model_dict.keys()))


def load_model(
    model_str="",
    attach_head=False,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    """
    Loads an architecture from the list of available architectures.
    
    model_str (str): Name of the model to be loaded. Use `list_models()` to view all available models.
    attach_head (bool): Whether to attach a classification (or other) head to the model. Default is False.
    num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
    dropout (float): Dropout value. Default is 0.
    pretrained: Whether to load pretrained weights or not. Default is False
    download_dir: The directory where the model weights are downloaded to. If not provided, the weights will be saved to `~/jax_models`.
    **kwargs: Any other parameters that are to be passed during model creation.
    
    """
    return model_dict[model_str](
        attach_head,
        num_classes,
        dropout=dropout,
        pretrained=pretrained,
        download_dir=download_dir,
        **kwargs
    )
