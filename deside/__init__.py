r"""EMT Decode"""


import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("deside").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.1-dev"
