from flax.serialization import msgpack_serialize, msgpack_restore
from flax.core import freeze

from tqdm import tqdm
import requests
import os


def download_checkpoint_params(url="", download_dir=None):
    if not download_dir:
        download_dir = os.path.join("~", "jax_models")

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    file_path = os.path.join(download_dir, url.split("/")[-1])

    if os.path.exists(file_path):
        print("Using cached weights instead of downloading again")
        return file_path

    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(file_path, "wb+") as file, tqdm(
        desc=file_path, total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=4096):
            size = file.write(data)
            bar.update(size)

    return file_path


def save_trained_params(params, file=""):
    with open(file, "wb+") as f:
        serialized = msgpack_serialize(params.unfreeze())
        f.write(serialized)
    print(f"Saved successfully to {file}")


def load_trained_params(file=""):
    print(file)
    with open(file, "rb") as f:
        content = f.read()
        restored_params = msgpack_restore(content)

    return freeze(restored_params)
