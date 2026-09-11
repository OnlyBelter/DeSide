from deside.plot.evaluate_result import _get_legend_handles


def test_get_legend_handles_supports_new_matplotlib_attribute():
    class NewLegend:
        legend_handles = ["a", "b"]

    assert _get_legend_handles(NewLegend()) == ["a", "b"]


def test_get_legend_handles_supports_legacy_matplotlib_attribute():
    class LegacyLegend:
        legendHandles = ["x"]

    assert _get_legend_handles(LegacyLegend()) == ["x"]
