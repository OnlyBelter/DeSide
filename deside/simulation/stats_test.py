import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from ..utility import print_df
sns.set()


def split_and_shuffle(exp_df, n):
    """
    Yield successive n-sized split_and_shuffle from a dataframe.
    https://stackoverflow.com/a/48791962/2803344
    :param exp_df: pandas.DataFrame
        expression profile, genes x samples
    :param n: how many samples in each subgroup
    :return: a list of subgroup dataframe
    """
    sample_names = exp_df.columns.to_list()
    np.random.shuffle(sample_names)
    split_l = []
    n_subgroup = []
    for i in range(0, len(sample_names), n):
        sub_group_sample_names = sample_names[i:i+n]
        n_subgroup.append(len(sub_group_sample_names))
        split_l.append(exp_df.loc[:, sub_group_sample_names])
    print('>>> There are {} subgroups, each subgroup has {} samples.'.format(len(split_l),
                                                                             ', '.join([str(i) for i in n_subgroup])))
    return split_l


def two_group_ttest(group_a, group_b):
    """
    T-test between two groups
    :param group_a: pd.DataFrame
        group A, genes x samples
    :param group_b: pd.DataFrame
        group B, genes x samples
    :return: p-value of each gene between two groups
    """
    assert np.all(group_a.index == group_b.index), 'The order of genes in two groups should be same'
    result = pd.DataFrame(index=group_a.index, columns=['pvalue'])
    _, pvalue = stats.ttest_ind(group_a, group_b, axis=1, equal_var=False)
    result['pvalue'] = np.round(pvalue, 4)
    return result


def test_normality(exp_df):
    """
    test the normality of each gene across all samples by Shapiro test
    p >= 0.05 obey normal distribution
    :param exp_df: genes x samples
    :return:
    """
    exp_normality = pd.DataFrame(index=exp_df.index)
    print_df(exp_df)
    for gene in exp_df.index:
        gene_exp = exp_df.loc[gene, :]
        shapiro_test = stats.shapiro(gene_exp.values)
        # exp_normality.loc[gene, 'Shapiro_Tn'] = shapiro_test.statistic
        exp_normality.loc[gene, 'pvalue'] = shapiro_test.pvalue
    return exp_normality


def alpha_confidence_interval(gene_exp, alpha=0.05):
    """
    confidence interval (CI) of 1 - alpha in normal distribution
    :param gene_exp: a single gene expression across all samples
    :param alpha: significance level
    :return: CI
    """
    norm_dis = stats.norm(np.mean(gene_exp), np.std(gene_exp))
    min_ci = np.max([0, norm_dis.isf(1-alpha/2)])
    max_ci = norm_dis.isf(alpha/2)
    return [np.round(i, 3) for i in [min_ci, max_ci]]


if __name__ == '__main__':
    my_data = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    # print(split_and_shuffle(my_data, 4))
