import os
# import seaborn as sns
from ..utility import check_dir, set_fig_style
import matplotlib.pyplot as plt
# sns.set(palette='muted', font_scale=1.5)
set_fig_style()


def plot_sample_distribution(re_sampled_df, marker_ratio_tcga, marker_ratio_simu_bulk,
                             cell_type, bins=10, density: bool = True, fig_dir=None, mark_name=''):
    """
    plot the marker ratio distribution of a single cell type in a simulated bulk cell dataset
    :param re_sampled_df: marker ratio of each cell type after resampled
    :param marker_ratio_tcga: marker gene ratios of each cell type in TCGA
    :param marker_ratio_simu_bulk: marker ratio of each cell type before resampled
    :param cell_type: current cell type
    :param bins:
    :param density:
    :param fig_dir:
    :param mark_name: only for naming .png file
    :return:
    """
    if re_sampled_df is None:
        fig, ax = plt.subplots(2, 1, sharex='col', figsize=(10, 10))
        ax[0].hist(marker_ratio_tcga[cell_type], bins=bins, density=density, label='TCGA')
        ax[1].hist(marker_ratio_simu_bulk[cell_type], bins=bins, density=density, label='Before sampling')
        for j in range(2):
            ax[j].legend(loc='upper right')
        fig_name = f'marker_gene_ratio_{cell_type}_without_filtering{mark_name}.png'
    else:
        fig, ax = plt.subplots(2, 1, sharex='col', figsize=(10, 10))
        ax[0].hist(marker_ratio_tcga[cell_type], bins=bins, density=density, label='TCGA')
        # ax[1].hist(marker_ratio_simu_bulk[cell_type], bins=bins, density=density, label='Before sampling')
        ax[1].hist(re_sampled_df[cell_type], bins=bins, density=density, label='After sampling')
        for j in range(2):
            ax[j].legend(loc='upper right')
        fig_name = f'marker_gene_ratio_{cell_type}_with_filtering{mark_name}.png'

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Marker gene ratio")
    plt.ylabel("Density")
    plt.title(f'Marker gene ratio of {cell_type} in simu bulk')
    # plt.legend()
    plt.tight_layout()

    if fig_dir is not None:
        check_dir(fig_dir)
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=200)
    plt.close()
