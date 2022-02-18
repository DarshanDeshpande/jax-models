import fnmatch
import re

model_dict = {}


def register_model(function):
    model_name = function.__name__
    model_name = re.sub("_", "-", model_name.lower())
    model_dict[model_name] = function
    return function


def list_models(filter=""):
    """
    Lists all available model architectures.

    filter (str): Pattern to match while listing models. Example: swin* will match all swin models and swin-large-384 will match the specific model.
    """
    if filter:
        if "*" not in filter:
            filter = "".join([filter, "*"])

        model_list = []
        for model in model_dict.keys():
            if fnmatch.fnmatch(model, filter):
                model_list.append(model)

        return sorted(model_list)
    else:
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
