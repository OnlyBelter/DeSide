r"""EMT Decode"""


import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("deside").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.1-dev"


def predict_with_pretrained_model(*args, **kwargs):
    from .decon_cf import predict_with_pretrained_model as _predict_with_pretrained_model
    return _predict_with_pretrained_model(*args, **kwargs)
