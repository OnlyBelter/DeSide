from .pub_func import print_df
from .pub_func import filter_gene_by_expression_log_mean, filter_gene_by_expression_min_max, filter_sample_by_expression
from .pub_func import log2_transform, center_value
from .pub_func import filter_gene_by_variance
from .pub_func import read_exp_from_hcluster
from .pub_func import read_cancer_purity
from .pub_func import cal_relative_error
from .pub_func import calculate_rmse, calculate_r2
from .pub_func import check_dir, parse_log_file, write_to_log
from .pub_func import correct_gene_list
from .pub_func import extract_gz_file
from .pub_func import read_data_from_h5ad, create_h5ad_dataset
from .pub_func import log_exp2cpm, ciber_exp, non_log2log_cpm, non_log2cpm
from .pub_func import get_corr, get_sep
from .pub_func import read_marker_gene
from .pub_func import cal_exp_by_gene_list
from .pub_func import print_msg, save_key_params
from .pub_func import read_xy, read_df
from .pub_func import cal_corr_gene_exp_with_cell_frac
from .pub_func import aggregate_marker_gene_exp
from .pub_func import cal_marker_ratio
from .core_obj import QueryNeighbors, ExpObj
from .pub_func import default_core_marker_genes
from .pub_func import get_cell_num
from .pub_func import sorted_cell_types, get_inx2cell_type
from .pub_func import do_pca_analysis, do_umap_analysis
from .pub_func import set_fig_style
from .evaluation import get_core_zone_of_pca
from .pub_func import get_ccc


subcell_type2abbr = {'B Cells (1)': 'B1', 'B Cells (2)': 'B2', 'B Cells (3)': 'B3', 'B Cells (4)': 'B4',
                     'Epithelial Cells (1)': 'CC1', 'Epithelial Cells (2)': 'CC2', 'Epithelial Cells (3)': 'CC3',
                     'CD4 T (1)': 'CD4T1', 'CD4 T (2)': 'CD4T2', 'CD4 T (3)': 'CD4T3',
                     'CD8 T': 'CD8T', 'CD8 Tex': 'CD8Tex',
                     'pDC': 'pDC', 'mDC': 'mDC', 'Fibroblasts (1)': 'F1', 'Fibroblasts (2)': 'F2',
                     'Mast Cells': 'MC', 'NK': 'NK',
                     'Macrophages (1)': 'MP1', 'Macrophages (2)': 'MP2',
                     'Macrophages (3)': 'MP3', 'Macrophages (4)': 'MP4',
                     'Neutrophils': 'NEU', 'Endothelial Cells': 'EC'}


cell_type2abbr = {'B Cells': 'B', 'CD4 T': 'CD4T', 'CD8 T': 'CD8T', 'Cancer Cells': 'CC', 'DC': 'DC',
                  'Endothelial Cells': 'EC', 'Fibroblasts': 'F', 'Macrophages': 'MP', 'Mast Cells': 'MC',
                  'NK': 'NK', 'Neutrophils': 'NEU'}

cell_type_mapping = {'B Cells (1)': 'B Cells', 'B Cells (2)': 'B Cells', 'B Cells (3)': 'B Cells',
                     'B Cells (4)': 'B Cells',
                     'Cancer Cells (1)': 'Cancer Cells', 'Cancer Cells (2)': 'Cancer Cells',
                     'Epithelial Cells (1)': 'Cancer Cells', 'Epithelial Cells (2)': 'Cancer Cells',
                     'Epithelial Cells (3)': 'Cancer Cells',
                     'CD4 T (1)': 'CD4 T', 'CD4 T (2)': 'CD4 T', 'CD4 T (3)': 'CD4 T',
                     'CD8 T': 'CD8 T', 'CD8 T (1)': 'CD8 T', 'CD8 T (2)': 'CD8 T', 'CD8 Tex': 'CD8 T',
                     'pDC': 'DC', 'mDC': 'DC',
                     'Fibroblasts': 'Fibroblasts', 'Fibroblasts (1)': 'Fibroblasts', 'Fibroblasts (2)': 'Fibroblasts',
                     'Mast Cells': 'Mast Cells', 'NK': 'NK',
                     'Macrophages': 'Macrophages', 'Macrophages (1)': 'Macrophages', 'Macrophages (2)': 'Macrophages',
                     'Macrophages (3)': 'Macrophages', 'Macrophages (4)': 'Macrophages',
                     'Monocyte/Macrophages (1)': 'Macrophages',
                     'Monocytes/Macrophages (2)': 'Macrophages',
                     'Neutrophils': 'Neutrophils', 'Endothelial Cells': 'Endothelial Cells'}


cancer_types = ['ACC', 'BLCA', 'BRCA', 'GBM', 'HNSC', 'LGG', 'LIHC', 'LUAD', 'PAAD', 'PRAD',
                'CESC', 'COAD', 'KICH', 'KIRC', 'KIRP', 'LUSC', 'READ', 'THCA', 'UCEC']


# used in the function "get_purified_gep" of decon_cf.reg_model and plot.evaluate_result
error_type = {2: 'both less', 3: 'z_score outlier', 7: 'relative error outlier', 8: 'both greater'}

# core marker genes, ref to file "DeSide_example/01single_cell/selected_marker_genes.csv"
default_core_marker_genes_old = {'CD4 T': ['MAL', 'CD40LG', 'FBLN7', 'IL7R', 'FOXP3'],
                                 'CD8 T': ['CD8A', 'CD8B'], 'T Cells': ['CD2', 'CD3D', 'CD3E'],
                                 'B Cells': ['VPREB3', 'POU2AF1', 'BANK1', 'CD19', 'CD22', 'CD79A', 'FCRL5', 'MS4A1'],
                                 'DC': ['LAMP3', 'FLT3', 'IRF7', 'PPP1R14B', 'SMPD3'],
                                 'Endothelial Cells': ['CLDN5', 'ENG', 'PLVAP', 'VWF'],
                                 'Fibroblasts': ['COL1A1', 'COL1A2', 'COL3A1', 'MYL9'],
                                 'Macrophages': ['AIF1', 'CD14', 'CD163', 'MS4A7', 'C1QC', 'C1QB'],
                                 'Mast Cells': ['CPA3', 'HPGDS', 'GATA2'],
                                 'NK': ['KLRD1', 'CEP78'],
                                 'Neutrophils': ['CSF3R', 'CXCR2', 'FPR1']}


inx2single_cell_dataset_id = {
    1: 'hnscc_cillo_01',
    2: 'pdac_pengj_02',
    3: 'hnscc_puram_03',
    4: 'pdac_steele_04',
    5: 'luad_kim_05',
    6: 'nsclc_guo_06',
    7: 'pan_cancer_07'
}
