r"""EMT Decode"""


def _resolve_version():
    try:
        from importlib.metadata import version as _pkg_version
    except ImportError:
        # Python 3.7 fallback; DeSide's floor is 3.9, so this is defensive
        try:
            from pkg_resources import get_distribution
            return get_distribution("deside").version
        except Exception:
            return "0.1-dev"
    try:
        return _pkg_version("deside")
    except Exception:
        return "0.1-dev"


__version__ = _resolve_version()


def predict_with_pretrained_model(*args, **kwargs):
    from .decon_cf import predict_with_pretrained_model as _predict_with_pretrained_model
    return _predict_with_pretrained_model(*args, **kwargs)
