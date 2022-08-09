import os
import time
import scipy
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.spatial import distance
from scipy.cluster import hierarchy
from ..utility import print_df, log2_transform, center_value, set_fig_style
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

set_fig_style()


def plot_hcluster(exp_df, sample2subtype=None, method='sample_corr',
                  color_threshold=4.0, result_dir='', sample_type='', p=8,
                  leaf_font_size=6, min_sample_in_cluster=15, log_transform=True, center=True):
    """
    plot heatmap and dendrogram after hierarchy clustering
    :param exp_df: expression dataframe, gene x samples, non-log and non-centered values
        - can be centered by genes after log2 normalized for calculating sample correlation
        - or clustering by gene expression directly
    :param sample2subtype: dataframe, sample x subtype
        sample annotation, must contain 'subtype' column and use sample name as index
    :param method: str
        sample_corr (correlation between each sample) or distance (between genes <row-wise> and samples <col-wise>)
    :param color_threshold: color threshold for dendrogram graph
    :param result_dir:
    :param sample_type: only for naming result file or naming subtype if sample2subtype is None
    :param p:  int, optional
        The ``p`` parameter for ``truncate_mode`` in hierarchy.dendrogram.
    :param leaf_font_size: leaf_font_size
    :param min_sample_in_cluster: if the number of samples in a cluster < this number, this cluster will be removed
    :param log_transform: if logarithm
    :param center: if center data
    :return: reordered samples in each cluster and cluster id
    """
    if sample2subtype is None:
        sample2subtype = pd.DataFrame(index=exp_df.columns, columns=['subtype'])
        sample2subtype['subtype'] = sample_type
    else:
        assert 'subtype' in sample2subtype.columns
    sample2subtype = sample2subtype.loc[:, ['subtype']].copy()
    subtype = np.unique(list(sample2subtype['subtype']))
    _colors = [i for i in mcolors.TABLEAU_COLORS.keys()]
    if len(subtype) > 10:
        print('There are more than 10 subtypes, some colors may used more than 1 time.')
        _colors = _colors * (len(subtype)//10 + 1)
    color = _colors[:len(subtype)]
    lut = dict(zip(subtype, color))
    print('subtype2color: {}'.format(lut))
    row_colors = sample2subtype['subtype'].map(lut)
    if log_transform:
        exp_df = log2_transform(exp_df)

    if method == 'sample_corr':
        # center log-transformed data before calculate correlation
        if center:
            exp_df = center_value(exp_df)
        sample_corr = exp_df.corr()
        row_linkage = hierarchy.linkage(
            distance.pdist(sample_corr), method='centroid')

        col_linkage = hierarchy.linkage(
            distance.pdist(sample_corr.T), method='centroid')
        if len(subtype) >= 2:
            g1 = sns.clustermap(sample_corr,
                                row_linkage=row_linkage, col_linkage=col_linkage,
                                row_colors=row_colors, col_colors=row_colors,
                                method="centroid", figsize=(15, 15), cmap='vlag')
        else:
            g1 = sns.clustermap(sample_corr,
                                row_linkage=row_linkage, col_linkage=col_linkage,
                                method="centroid", figsize=(15, 15), cmap='vlag')
    else:
        # distance between gene pairs, treat each row as a sample
        # row_linkage = hierarchy.linkage(
        #     distance.pdist(exp_df), metric='euclidean', method='centroid')

        # distance between sample pairs
        col_linkage = hierarchy.linkage(
            distance.pdist(exp_df.T), metric='euclidean', method='centroid')
        g1 = sns.clustermap(exp_df, row_linkage=None, col_linkage=col_linkage,
                            col_colors=row_colors, z_score=0,
                            method="centroid", figsize=(18, 18), cmap='vlag')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    g1.savefig(os.path.join(result_dir, '{}_hierarchy_clustering.png'.format(sample_type)), dpi=200)
    plt.close()

    plt.figure(figsize=(18, 6))
    hierarchy.dendrogram(col_linkage, show_leaf_counts=True, p=p, truncate_mode='level', leaf_font_size=leaf_font_size,
                         leaf_rotation=50, color_threshold=color_threshold)
    plt.savefig(os.path.join(result_dir, '{}_dendrogram.png'.format(sample_type)), dpi=200)
    plt.close()
    # return {'row_linkage': row_linkage, 'col_linkage': col_linkage, 'g1': g1}

    # get sub cluster info: reordered samples in each cluster and cluster id
    node_id2sub_cluster = cut_cluster_by_distance(col_linkage, threshold=color_threshold)
    reordered_ind2sample_name = {}
    # g1 = linkage['g1']
    # sample_corr = bulk_tpm_log_c.corr()
    for ind in g1.dendrogram_col.reordered_ind:
        reordered_ind2sample_name[ind] = exp_df.columns[ind]
    reordered_ind2sample_name_df = pd.DataFrame.from_dict(data=reordered_ind2sample_name, orient='index',
                                                          columns=['sample_name'])
    # reordered_ind2sample_name_df.head(2)
    reordered_ind2sample_name_df['cluster_id'] = reordered_ind2sample_name_df.index.map(node_id2sub_cluster)
    reordered_ind2sample_name_df2 = reordered_ind2sample_name_df.merge(sample2subtype, left_on='sample_name',
                                                                       right_index=True)
    reordered_ind2sample_name_df2.index.name = 'reordered_inx'
    print_df(reordered_ind2sample_name_df2)
    reordered_ind2sample_name_df2['color'] = reordered_ind2sample_name_df2['subtype'].map(lut)

    subtype_pivot = reordered_ind2sample_name_df2.pivot_table(index=['cluster_id'], columns='subtype',
                                                              aggfunc='size', fill_value=0)
    non_new = subtype_pivot.loc[:, [i for i in subtype_pivot.columns if i != 'New']]
    max_inx = np.where(non_new == np.vstack(non_new.max(axis=1)))
    cluster_id2max_col_inx = dict(zip(max_inx[0], max_inx[1]))  # prevent multiple values equal to max
    subtype_pivot.columns = subtype_pivot.columns.astype(str)
    subtype_pivot['predicted_subtype'] = [subtype_pivot.columns[cluster_id2max_col_inx[x]]
                                          for x in range(subtype_pivot.shape[0])]
    # subtype_pivot['predicted_subtype'] = np.vstack(
    #     non_new.columns[np.where(non_new == np.vstack(non_new.max(axis=1)))[1]].to_list())
    _main_subtype_percent_name = 'main_subtype%'
    if 'New' in subtype_pivot.columns:
        _main_subtype_percent_name = 'main_subtype_with_new%'
        subtype_pivot[_main_subtype_percent_name] = (non_new.max(axis=1) + subtype_pivot['New']) / subtype_pivot.sum(axis=1)

    else:
        subtype_pivot[_main_subtype_percent_name] = non_new.max(axis=1) / subtype_pivot.sum(axis=1)
        # subtype_pivot['keep'] = (non_new.max(axis=1) >= 5) &
    subtype_pivot['keep'] = (non_new.max(axis=1) >= min_sample_in_cluster) & \
                            (subtype_pivot[_main_subtype_percent_name] >= 0.8)
    subtype_pivot['keep'] = subtype_pivot['keep'].map({True: 1, False: 0})
    reordered_ind2sample_name_df2 = reordered_ind2sample_name_df2.merge(subtype_pivot[['predicted_subtype',
                                                                                       _main_subtype_percent_name,
                                                                                       'keep']],
                                                                        left_on='cluster_id', right_index=True)

    subtype_pivot.to_csv(os.path.join(result_dir, 'subtype_pivot.csv'))
    reordered_ind2sample_name_df2.to_csv(os.path.join(result_dir, 'reordered_ind2sample_name.csv'),
                                         index=False, float_format='%.3f')
    with open(os.path.join(result_dir, 'color_threshold.txt'), 'w') as f:
        f.write(str(color_threshold) + '\n')
    return reordered_ind2sample_name_df2


def split_tree_by_distance(node, threshold):
    assert type(node) == scipy.cluster.hierarchy.ClusterNode
    if node.count < 2:
        return [node]
    else:
        if node.dist <= threshold:
            return [node]
        else:
            return split_tree_by_distance(node.left, threshold) + split_tree_by_distance(node.right, threshold)


def cut_cluster_by_distance(linkage_matrix, threshold):
    """
    linkage_matrix: the matrix comes from hierarchy.linkage,
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    threshold: a float, max distance of clusters in the tree at the cut point
    """
    node_id2sub_cluster = {}
    root_node = hierarchy.to_tree(linkage_matrix)
    root_node_in_sub_clusters = split_tree_by_distance(root_node, threshold=threshold)
    for i, sub_rootnodes in enumerate(root_node_in_sub_clusters):
        subcluster_nodes = get_node_id(sub_rootnodes)
        for _node_id in subcluster_nodes:
            node_id2sub_cluster[_node_id] = i
    return node_id2sub_cluster


def get_node_id(node):
    assert type(node) == scipy.cluster.hierarchy.ClusterNode
    if node.count == 1:
        return [node.id]
    else:
        left_node = node.left
        right_node = node.right
        return get_node_id(left_node) + get_node_id(right_node)


def t_sne_plot(exp_profile, group2sample, result_file=None, n_jobs=4,
               n_iter=3000, perplexity=10):
    """
    plot expression profiles by t-SNE
    :param exp_profile: a dataFrame, gene x sample
        the expression profiles of single cell or bulk cell
    :param group2sample: dict
        cell type with all sample names in this cell type, {'': [], '': [], ...}
    :param result_file: where to save figure
    :param n_jobs:
    :param n_iter:
    :param perplexity:
    :return:
    """
    t0 = time.time()
    exp_profile = exp_profile.T  # convert to sample by gene
    tsne = TSNE(n_components=2, n_jobs=n_jobs, learning_rate=200, metric='euclidean',
                n_iter=n_iter, init='pca', verbose=1, perplexity=perplexity)
    x_reduced = tsne.fit_transform(exp_profile)
    # X_reduced_tsne = tsne.fit(x)
    print(x_reduced.shape)
    # np.save('X_reduced_tsne_pca_first', X_reduced_tsne2)
    t1 = time.time()
    print("t-SNE took {:.1f}s.".format(t1 - t0))
    x_reduced = pd.DataFrame(data=x_reduced, index=exp_profile.index)
    # groups = np.unique(list(group2sample.values()))
    plt.figure(figsize=(8, 6))
    for g, current_samples in group2sample.items():
        # current_samples = [i for i, j in group2sample.items() if j == g]
        current_corr = x_reduced.loc[x_reduced.index.isin(current_samples), :]
        plt.scatter(current_corr[0], current_corr[1], label=g, s=6)
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.legend()
    plt.tight_layout()
    if result_file:
        plt.savefig(result_file, dpi=200)
        plt.close()
        x_reduced.to_csv(result_file.replace('.png', '.csv'))
    else:
        return plt
