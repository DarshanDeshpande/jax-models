"""
A major part of this code is translated from https://github.com/facebookresearch/mae
"""

import jax.numpy as jnp

import flax.linen as nn
from ..layers import Attention, DropPath, TransformerMLP, Mask, PatchEmbed
from .model_registry import register_model

from typing import Optional
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

__all__ = ["MaskedAutoencoderViT", "mae_base", "mae_large", "mae_huge"]


def sincosemb1d(emb_dim, pos):
    omega = jnp.arange(emb_dim // 2, dtype=jnp.float32)
    omega /= emb_dim / 2.0
    omega = 1.0 / 10000 * omega

    pos = pos.reshape(-1)
    out = jnp.einsum("m, d -> md", pos, omega)

    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)


def sincosposemb(emb_dim, grid_size):
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)
    grid = jnp.stack(grid, axis=0)
    grid = jnp.reshape(grid, [2, 1, grid_size, grid_size])

    emb_height = sincosemb1d(emb_dim // 2, grid[0])
    emb_width = sincosemb1d(emb_dim // 2, grid[1])

    pos_emb = jnp.concatenate([emb_height, emb_width], axis=1)
    return jnp.expand_dims(jnp.concatenate([jnp.zeros([1, emb_dim]), pos_emb], 0), 0)


pos_emb_init = sincosposemb
emb_init = nn.initializers.normal(stddev=0.02)


class Block(nn.Module):
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    att_drop: float = 0.1
    drop: float = 0.1
    drop_path: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        x = nn.LayerNorm()(inputs)
        x = Attention(self.dim, self.num_heads, True, self.att_drop, self.drop)(
            x, deterministic
        )
        x = DropPath(self.drop_path)(x, deterministic)
        inputs = inputs + x

        x = nn.LayerNorm()(inputs)
        x = TransformerMLP(
            self.dim * self.mlp_ratio, self.dim, self.drop, use_dwconv=False
        )(x, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        return inputs + x


class Encoder(nn.Module):
    patch_size: int = 16
    emb_dim: int = 1024
    depth: int = 24
    mask_ratio: float = 0.5
    att_drop: float = 0
    drop: float = 0
    drop_path: float = 0
    num_heads: int = 16
    mlp_ratio: int = 4
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, patches, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        patch_embed = PatchEmbed(self.patch_size, self.emb_dim, use_norm=False)(patches)
        num_patches = patch_embed.shape[1]

        pos_emb = self.variable(
            "pos_emb",
            "enc_pos_emb",
            pos_emb_init,
            self.emb_dim,
            int(num_patches ** 0.5),
        )

        x = patch_embed + pos_emb.value[:, 1:, :]
        x, mask, ids_restore = Mask()(x, self.mask_ratio)

        cls_token = self.param("cls_token", emb_init, (1, 1, self.emb_dim))
        cls_token = cls_token + pos_emb.value[:, :1, :]
        cls_token = jnp.broadcast_to(
            cls_token, (x.shape[0], cls_token.shape[1], cls_token.shape[2])
        )
        x = jnp.concatenate([cls_token, x], axis=1)

        for _ in range(self.depth):
            x = Block(
                self.emb_dim,
                self.num_heads,
                self.mlp_ratio,
                self.att_drop,
                self.drop,
                self.drop_path,
            )(x, deterministic)

        x = nn.LayerNorm()(x)

        return x, mask, ids_restore, num_patches


class Decoder(nn.Module):
    patch_size: int = 16
    dec_emb_dim: int = 512
    dec_depth: int = 8
    num_patches: int = 196
    att_drop: float = 0
    drop: float = 0
    drop_path: float = 0
    dec_num_heads: int = 16
    mlp_ratio: int = 4
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, encoder_output, ids_restore, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        x = nn.Dense(self.dec_emb_dim)(encoder_output)

        mask_token = self.param("mask_token", emb_init, (1, 1, self.dec_emb_dim))
        mask_tokens = jnp.tile(
            mask_token, (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        )

        x_ = jnp.concatenate([x[:, 1:, :], mask_tokens], axis=1)
        x_ = jnp.take_along_axis(
            x_, jnp.tile(jnp.expand_dims(ids_restore, -1), (1, 1, x.shape[2])), axis=1
        )
        x = jnp.concatenate([x[:, :1, :], x_], axis=1)

        decoder_pos_emb = self.variable(
            "pos_emb",
            "dec_pos_emb",
            pos_emb_init,
            self.dec_emb_dim,
            int(self.num_patches ** 0.5),
        )

        x = x + decoder_pos_emb.value

        for _ in range(self.dec_depth):
            x = Block(
                self.dec_emb_dim,
                self.dec_num_heads,
                self.mlp_ratio,
                self.att_drop,
                self.drop,
                self.drop_path,
            )(x, deterministic)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.patch_size ** 2 * 3)(x)

        return x[:, 1:, :]


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder Module

    Attributes:
        patch_size (int): Patch size. Default is 16.
        emb_dim (int): Embedding dimension. Default is 1024.
        dec_emb_dim (int): Decoder embedding dimension. Default is 512.
        depth (int): Depth for encoder block.
        dec_depth (int): Depth for decoder block.
        mask_ratio (float)[0,1]: Percentage of image to mask.
        att_drop (float): Dropout value for attention Default is 0.
        drop (float): Dropout value. Default is 0.
        drop_path (float): Dropout value for DropPath. Default is 0.
        num_heads (int): Number of attention heads. Default is 16.
        dec_num_heads (int): Number of decoder attention heads. Default is 16.
        mlp_ratio (int): Multiplier for hidden dimension in transformer MLP block. Default is 4.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    patch_size: int = 16
    emb_dim: int = 1024
    dec_emb_dim: int = 512
    depth: int = 24
    dec_depth: int = 8
    mask_ratio: float = 0.5
    att_drop: float = 0
    drop: float = 0
    drop_path: float = 0
    num_heads: int = 16
    dec_num_heads: int = 16
    mlp_ratio: int = 4
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        x, mask, ids_restore, num_patches = Encoder(
            self.patch_size,
            self.emb_dim,
            self.depth,
            self.mask_ratio,
            self.att_drop,
            self.drop,
            self.drop_path,
            self.num_heads,
            self.mlp_ratio,
        )(inputs, deterministic)

        x = Decoder(
            self.patch_size,
            self.dec_emb_dim,
            self.dec_depth,
            num_patches,
            self.att_drop,
            self.drop,
            self.drop_path,
            self.dec_num_heads,
            self.mlp_ratio,
        )(x, ids_restore, deterministic)

        return x, mask


@register_model
def mae_base(
    attach_head=None,
    num_classes=None,
    dropout=None,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    logging.info(
        "Default classification arguments are ignored since this is a generative model. To tune the hyperparameters, please call `MaskedAutoencoderViT` separately with your desired arguments."
    )
    del attach_head, num_classes, dropout

    if pretrained:
        logging.info("Pretrained MAE Base isn't available. Loading un-trained model.")

    return MaskedAutoencoderViT(
        patch_size=16,
        emb_dim=768,
        depth=12,
        num_heads=12,
        dec_emb_dim=512,
        dec_depth=8,
        dec_num_heads=16,
        mlp_ratio=4,
        **kwargs
    )


@register_model
def mae_large(
    attach_head=None,
    num_classes=None,
    dropout=None,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    logging.info(
        "Default classification arguments are ignored since this is a generative model. To tune the hyperparameters, please call `MaskedAutoencoderViT` separately with your desired arguments."
    )
    del attach_head, num_classes, dropout

    if pretrained:
        logging.info("Pretrained MAE Large isn't available. Loading un-trained model.")

    return MaskedAutoencoderViT(
        patch_size=16,
        emb_dim=1024,
        depth=24,
        num_heads=16,
        dec_emb_dim=512,
        dec_depth=8,
        dec_num_heads=16,
        mlp_ratio=4,
        **kwargs
    )


@register_model
def mae_huge(
    attach_head=None,
    num_classes=None,
    dropout=None,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    logging.info(
        "Default classification arguments are ignored since this is a generative model. To tune the hyperparameters, please call `MaskedAutoencoderViT` separately with your desired arguments."
    )
    del attach_head, num_classes, dropout

    if pretrained:
        logging.info("Pretrained MAE Huge isn't available. Loading un-trained model.")

    return MaskedAutoencoderViT(
        patch_size=14,
        emb_dim=1280,
        depth=32,
        num_heads=16,
        dec_emb_dim=512,
        dec_depth=8,
        dec_num_heads=16,
        mlp_ratio=4,
        **kwargs
    )
