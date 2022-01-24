from flax.serialization import msgpack_serialize, msgpack_restore
from flax.core import freeze


def save_trained_params(params, file=""):
    with open(file, "wb+") as f:
        serialized = msgpack_serialize(params.unfreeze())
        f.write(serialized)
    print(f"Saved successfully to {file}")


def load_trained_params(file=""):
    with open("swin_tiny_224.weights", "rb") as f:
        content = f.read()
        restored_params = msgpack_restore(content)

    return freeze(restored_params)
