import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    mlp_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.mlp_dim, kernel_init=nn.initializers.xavier_uniform())(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


class TransformerEncoder(nn.Module):
    mlp_dim: int
    pool_size: int
    stride: int

    @nn.compact
    def __call__(self, inputs):
        norm = nn.LayerNorm()(inputs)
        att = nn.avg_pool(
            norm,
            (self.pool_size, self.pool_size),
            strides=(self.stride, self.stride),
            padding="SAME",
        )
        att = att - norm
        add = inputs + att
        x = nn.LayerNorm()(add)
        x = MLP(self.mlp_dim, self.mlp_dim)(x)
        return add + x


class AddPositionEmbs(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2], inputs.shape[3])
        pe = self.param(
            "pos_embedding", nn.initializers.normal(stddev=0.02), pos_emb_shape
        )
        return inputs + pe


class S12(nn.Module):
    attach_head: bool = False
    num_classes: int = 1000

    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(64, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(2):
            x = TransformerEncoder(64, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(128, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(2):
            x = TransformerEncoder(128, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(320, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(320, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(512, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(2):
            x = TransformerEncoder(512, 3, 1)(x)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)
        return x


class S24(nn.Module):
    attach_head: bool = False
    num_classes: int = 1000

    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(64, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(4):
            x = TransformerEncoder(64, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(128, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(4):
            x = TransformerEncoder(128, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(320, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(12):
            x = TransformerEncoder(320, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(512, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(4):
            x = TransformerEncoder(512, 3, 1)(x)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)

        return x


class S36(nn.Module):
    attach_head: bool = False
    num_classes: int = 1000

    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(64, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(64, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(128, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(128, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(320, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(18):
            x = TransformerEncoder(320, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(512, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(512, 3, 1)(x)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)

        return x


class M36(nn.Module):
    attach_head: bool = False
    num_classes: int = 1000

    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(96, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(96, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(192, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(192, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(384, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(18):
            x = TransformerEncoder(384, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(768, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(768, 3, 1)(x)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)

        return x


class M48(nn.Module):
    attach_head: bool = False
    num_classes: int = 1000

    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(96, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(8):
            x = TransformerEncoder(96, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(192, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(8):
            x = TransformerEncoder(192, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(384, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(24):
            x = TransformerEncoder(384, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(768, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(8):
            x = TransformerEncoder(768, 3, 1)(x)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)

        return x


def PoolFormer_S12(attach_head=False, num_classes=1000):
    return S12(attach_head, num_classes)


def PoolFormer_S24(attach_head=False, num_classes=1000):
    return S24(attach_head, num_classes)


def PoolFormer_S36(attach_head=False, num_classes=1000):
    return S36(attach_head, num_classes)


def PoolFormer_M36(attach_head=False, num_classes=1000):
    return M36(attach_head, num_classes)


def PoolFormer_M48(attach_head=False, num_classes=1000):
    return M48(attach_head, num_classes)
