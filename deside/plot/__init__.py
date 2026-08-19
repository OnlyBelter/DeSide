from .plot_nn import plot_loss, plot_paras, plot_paras_all_cell_types
from .plot_nn import plot_corr_two_columns, plot_predicted_result
from .plot_gene import compare_exp_between_group

try:
    from .plot_clustering import plot_hcluster, t_sne_plot
    from .plot_gene import (
        plot_single_gene_exp,
        plot_gene_pdf,
        plot_emt_gene_exp,
        plot_cd8_marker,
        plot_gene_exp,
        plot_marker_gene_in_cell_type,
        plot_marker_exp,
        plot_marker_ratio,
    )
    from .evaluate_result import (
        compare_y_y_pred_plot,
        compare_exp_and_cell_fraction,
        compare_cell_fraction_across_cancer_type,
        plot_pca,
        plot_clustermap,
        compare_mean_exp_with_cell_frac_across_algo,
        ScatterPlot,
        plot_pred_cell_prop_with_cpe,
        compare_y_y_pred_plot_cpe,
    )
    from .plot_sample import plot_sample_distribution
except ImportError:
    # Keep lightweight DeSide imports working even when optional plotting
    # dependencies are not installed in the active environment.
    pass
