from .generate_data import BulkGEPGenerator, SingleCellTypeGEPGenerator
from .generate_data import filtering_by_gene_list_and_pca_plot, get_gene_list_for_filtering, cal_loading_by_pca
from .generate_data import fragment_generation_fraction, random_generation_fraction, segment_generation_fraction

__all__ = [
    "BulkGEPGenerator",
    "SingleCellTypeGEPGenerator",
    "filtering_by_gene_list_and_pca_plot",
    "get_gene_list_for_filtering",
    "cal_loading_by_pca",
    "segment_generation_fraction",
    "random_generation_fraction",
    "fragment_generation_fraction",
    "split_and_shuffle",
    "test_normality",
    "two_group_ttest",
    "alpha_confidence_interval",
]


def split_and_shuffle(*args, **kwargs):
    from .stats_test import split_and_shuffle as _split_and_shuffle

    return _split_and_shuffle(*args, **kwargs)


def test_normality(*args, **kwargs):
    from .stats_test import test_normality as _test_normality

    return _test_normality(*args, **kwargs)


def two_group_ttest(*args, **kwargs):
    from .stats_test import two_group_ttest as _two_group_ttest

    return _two_group_ttest(*args, **kwargs)


def alpha_confidence_interval(*args, **kwargs):
    from .stats_test import alpha_confidence_interval as _alpha_confidence_interval

    return _alpha_confidence_interval(*args, **kwargs)
