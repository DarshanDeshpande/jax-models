import jax.numpy as jnp
import jax.random as random
import flax.linen as nn

from ..layers import DropPath, TransformerMLP, PatchEmbed
from .helper import download_checkpoint_params, load_trained_params
from .model_registry import register_model

from typing import Optional

__all__ = [
    "CaiT",
    "cait_xxs24_224",
    "cait_xxs24_384",
    "cait_xxs36_224",
    "cait_xxs36_384",
    "cait_xs24_384",
    "cait_s24_224",
    "cait_s24_384",
    "cait_s36_384",
    "cait_m36_384",
    "cait_m48_448",
]

pretrained_cfgs = {
    "cait-xxs24-224": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_xxs24_224.weights",
    "cait-xxs24-384": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_xxs24_384.weights",
    "cait-xxs36-224": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_xxs36_224.weights",
    "cait-xxs36-384": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_xxs36_384.weights",
    "cait-xs24-384": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_xs24_384.weights",
    "cait-s24-224": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_s24_224.weights",
    "cait-s24-384": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_s24_384.weights",
    "cait-s36-384": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_s36_384.weights",
    "cait-m36-384": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_m36_384.weights",
    "cait-m48-448": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.4-cait/cait_m48_448.weights",
}


class ClassAttention(nn.Module):
    dim: int
    num_heads: int
    att_drop: float = 0.0
    proj_dropout: float = 0.0
    use_att_bias: bool = False
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        batch, n, channels = inputs.shape
        q = nn.Dense(self.dim, use_bias=self.use_att_bias, name="q")(inputs[:, 0])
        q = jnp.expand_dims(q, 1).reshape(
            batch, 1, self.num_heads, channels // self.num_heads
        )
        q = jnp.transpose(q, (0, 2, 1, 3))
        q = q * ((self.dim // self.num_heads) ** -0.5)

        k = nn.Dense(self.dim, use_bias=self.use_att_bias, name="k")(inputs).reshape(
            batch, n, self.num_heads, channels // self.num_heads
        )
        k = jnp.transpose(k, (0, 2, 1, 3))

        v = nn.Dense(self.dim, use_bias=self.use_att_bias, name="v")(inputs)
        v = v.reshape(batch, n, self.num_heads, channels // self.num_heads)
        v = jnp.transpose(v, (0, 2, 1, 3))

        att = q @ jnp.swapaxes(k, -2, -1)
        att = nn.softmax(att, -1)
        att = nn.Dropout(self.att_drop, name="attn_drop")(att, deterministic)

        x_cls = jnp.swapaxes(att @ v, 1, 2).reshape(batch, 1, channels)
        x_cls = nn.Dense(self.dim, use_bias=self.use_att_bias, name="proj")(x_cls)
        x_cls = nn.Dropout(self.proj_dropout, name="proj_drop")(x_cls, deterministic)

        return x_cls


class LayerScaleBlockClassAttention(nn.Module):
    dim: int
    num_heads: int
    att_drop: float = 0.0
    drop: float = 0.0
    drop_path: float = 0.0
    init_values: float = 1e-4
    mlp_ratio: float = 4
    use_att_bias: bool = False
    deterministic: Optional[bool] = None

    def gamma_init(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)

    @nn.compact
    def __call__(self, x, x_cls, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        gamma1 = self.param("gamma_1", self.gamma_init, (self.dim,), self.init_values)
        gamma2 = self.param("gamma_2", self.gamma_init, (self.dim,), self.init_values)
        hidden_dim = int(self.dim * self.mlp_ratio)

        u = jnp.concatenate([x_cls, x], axis=1)

        norm = nn.LayerNorm(name="norm1")(u)
        att = ClassAttention(
            self.dim,
            self.num_heads,
            self.att_drop,
            self.drop,
            self.use_att_bias,
            name="attn",
        )(norm, deterministic)
        x_cls = x_cls + DropPath(self.drop_path)(gamma1 * att, deterministic)

        norm = nn.LayerNorm(name="norm2")(x_cls)
        mlp = TransformerMLP(hidden_dim, self.dim, dropout=self.drop, name="mlp")(
            norm, deterministic
        )
        x_cls = x_cls + DropPath(self.drop_path)(gamma2 * mlp, deterministic)

        return x_cls


class TalkingHeadAttention(nn.Module):
    dim: int
    num_heads: int
    att_drop: float = 0.0
    proj_dropout: float = 0.0
    use_att_bias: bool = False
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        batch, n, channels = inputs.shape
        qkv = nn.Dense(self.dim * 3, use_bias=self.use_att_bias, name="qkv")(inputs)
        qkv = qkv.reshape(batch, n, 3, self.num_heads, channels // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = qkv[0] * ((self.dim // self.num_heads) ** -0.5), qkv[1], qkv[2]

        att = q @ jnp.swapaxes(k, -2, -1)
        att = nn.Dense(self.num_heads, name="proj_l")(jnp.transpose(att, (0, 2, 3, 1)))
        att = jnp.transpose(att, (0, 3, 1, 2))
        att = nn.softmax(att, -1)

        att = nn.Dense(self.num_heads, name="proj_w")(jnp.transpose(att, (0, 2, 3, 1)))
        att = jnp.transpose(att, (0, 3, 1, 2))
        att = nn.Dropout(self.att_drop, name="attn_drop")(att, deterministic)

        x = jnp.swapaxes(att @ v, 1, 2).reshape(batch, n, channels)
        x = nn.Dense(self.dim, name="proj")(x)
        x = nn.Dropout(self.proj_dropout, name="proj_drop")(x, deterministic)

        return x


class LayerScaleBlock(nn.Module):
    dim: int
    num_heads: int
    att_drop: float = 0.0
    drop: float = 0.0
    drop_path: float = 0.0
    init_values: float = 1e-4
    mlp_ratio: float = 4
    use_att_bias: bool = False
    deterministic: Optional[bool] = None

    def gamma_init(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)

    @nn.compact
    def __call__(self, x, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        gamma1 = self.param("gamma_1", self.gamma_init, (self.dim,), self.init_values)
        gamma2 = self.param("gamma_2", self.gamma_init, (self.dim,), self.init_values)
        hidden_dim = int(self.dim * self.mlp_ratio)

        norm = nn.LayerNorm(name="norm1")(x)
        att = TalkingHeadAttention(
            self.dim,
            self.num_heads,
            self.att_drop,
            self.drop,
            self.use_att_bias,
            name="attn",
        )(norm, deterministic)
        x = x + DropPath(self.drop_path)(gamma1 * att, deterministic)

        norm = nn.LayerNorm(name="norm2")(x)
        mlp = TransformerMLP(hidden_dim, self.dim, dropout=self.drop, name="mlp")(
            norm, deterministic
        )
        x = x + DropPath(self.drop_path)(gamma2 * mlp, deterministic)

        return x


class CaiT(nn.Module):
    """
    Module for Class-Attention in Image Transformers

    Attributes:
        patch_size (int): Patch size. Default is 16.
        embed_dim (int): Embedding dimension. Default is 768.
        depth (int): Number of blocks. Default is 12.
        num_heads (int): Number of attention heads. Default is 12.
        mlp_ratio (int): Multiplier for hidden dimension in transformer MLP block. Default is 4.
        use_att_bias (bool): Whether to use bias for linear qkv projection. Default is True.
        drop (float): Dropout value. Default is 0.
        att_dropout (float): Dropout value for attention Default is 0.
        drop_path (float): Dropout value for DropPath. Default is 0.
        init_scale (float): Initialization scale used for gamma initialization. Default is 1e-4.
        depth_token_only (int): Number of blocks with cls_token and class attention. Default is 2.
        mlp_ratio_clstk (int):  Multiplier for hidden dimension in transformer MLP block with class attention. Default is 4.
        attach_head (bool): Whether to attach classification head. Default is True
        num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4.0
    use_att_bias: bool = True
    drop: float = 0.0
    att_drop: float = 0.0
    drop_path: float = 0.0
    init_scale: float = 1e-4
    depth_token_only: int = 2
    mlp_ratio_clstk: int = 4.0
    attach_head: bool = True
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        x = PatchEmbed(self.patch_size, self.embed_dim, name="patch_embed")(inputs)
        num_patches = (inputs.shape[1] // self.patch_size) * (
            inputs.shape[2] // self.patch_size
        )

        cls_token = self.param("cls_token", nn.zeros, (1, 1, self.embed_dim))
        pos_embed = self.param("pos_embed", nn.zeros, (1, num_patches, self.embed_dim))

        cls_tokens = jnp.broadcast_to(
            cls_token, (inputs.shape[0], cls_token.shape[1], cls_token.shape[2])
        )

        x = x + pos_embed
        x = nn.Dropout(self.drop, name="pos_drop")(x, deterministic)

        for i in range(self.depth):
            x = LayerScaleBlock(
                self.embed_dim,
                self.num_heads,
                self.att_drop,
                self.drop,
                self.drop_path,
                self.init_scale,
                self.mlp_ratio,
                self.use_att_bias,
                name=f"blocks{i}",
            )(x, deterministic)

        for i in range(self.depth_token_only):
            cls_tokens = LayerScaleBlockClassAttention(
                self.embed_dim,
                self.num_heads,
                0.0,
                0.0,
                0.0,
                self.init_scale,
                self.mlp_ratio_clstk,
                self.use_att_bias,
                name=f"blocks_token_only{i}",
            )(x, cls_tokens, deterministic)

        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = nn.LayerNorm(name="norm")(x)
        x = x[:, 0]

        if self.attach_head:
            x = nn.Dense(self.num_classes, name="head")(x)

        return x


@register_model
def cait_xxs24_224(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=4,
        init_scale=1e-5,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-xxs24-224"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params


@register_model
def cait_xxs24_384(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=4,
        init_scale=1e-5,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-xxs24-384"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params


@register_model
def cait_xxs36_224(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=192,
        depth=36,
        num_heads=4,
        init_scale=1e-5,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-xxs36-224"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params


@register_model
def cait_xxs36_384(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=192,
        depth=36,
        num_heads=4,
        init_scale=1e-5,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-xxs36-384"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params


@register_model
def cait_xs24_384(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=288,
        depth=24,
        num_heads=6,
        init_scale=1e-5,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-xs24-384"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params


@register_model
def cait_s24_224(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=384,
        depth=24,
        num_heads=8,
        init_scale=1e-5,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-s24-224"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params


@register_model
def cait_s24_384(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=384,
        depth=24,
        num_heads=8,
        init_scale=1e-5,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-s24-384"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params


@register_model
def cait_s36_384(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=384,
        depth=36,
        num_heads=8,
        init_scale=1e-6,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-s36-384"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params


@register_model
def cait_m36_384(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=768,
        depth=36,
        num_heads=16,
        init_scale=1e-6,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-m36-384"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params


@register_model
def cait_m48_448(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = CaiT(
        patch_size=16,
        embed_dim=768,
        depth=48,
        num_heads=16,
        init_scale=1e-6,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["cait-m48-448"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params
