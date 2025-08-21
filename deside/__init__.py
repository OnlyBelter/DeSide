r"""DeSide: Cellular Deconvolution of Bulk RNA-seq

A unified DEep-learning and SIngle-cell based DEconvolution method for solid tumors
"""

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("deside").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.1-dev"

# Main DeSide class - the core deconvolution model
try:
    from .decon_cf import DeSide
except ImportError:
    # If dependencies are not available, define a placeholder
    def DeSide(*args, **kwargs):
        raise ImportError("DeSide requires additional dependencies. Please install with: pip install deside[full]")

# Key data reading and processing classes
try:
    from .utility.read_file import ReadH5AD, ReadExp
    from .utility import (
        # Data preprocessing functions  
        filter_gene_by_expression_log_mean,
        filter_gene_by_expression_min_max,
        filter_sample_by_expression,
        filter_gene_by_variance,
        log2_transform,
        center_value,
        
        # Data format conversion functions
        log_exp2cpm,
        non_log2log_cpm, 
        non_log2cpm,
        ciber_exp,
        
        # File I/O functions
        read_data_from_h5ad,
        create_h5ad_dataset,
        read_df,
        
        # Evaluation functions
        calculate_rmse,
        calculate_r2,
        calculate_mae,
        cal_relative_error,
        
        # Utility functions
        check_dir,
        print_df,
        print_msg,
        
        # Cell type and marker functions
        default_core_marker_genes,
        read_marker_gene,
        sorted_cell_types,
        get_inx2cell_type,
        
        # Analysis functions
        do_pca_analysis,
        do_umap_analysis,
        get_corr,
        get_corr_spearman,
    )
except ImportError as e:
    # Define placeholder functions if utility imports fail
    def _import_error_placeholder(name):
        def placeholder(*args, **kwargs):
            raise ImportError(f"Function {name} requires additional dependencies. Please install with: pip install deside[full]")
        return placeholder
    
    ReadH5AD = _import_error_placeholder("ReadH5AD")
    ReadExp = _import_error_placeholder("ReadExp")
    filter_gene_by_expression_log_mean = _import_error_placeholder("filter_gene_by_expression_log_mean")
    filter_gene_by_expression_min_max = _import_error_placeholder("filter_gene_by_expression_min_max")
    filter_sample_by_expression = _import_error_placeholder("filter_sample_by_expression")
    filter_gene_by_variance = _import_error_placeholder("filter_gene_by_variance")
    log2_transform = _import_error_placeholder("log2_transform")
    center_value = _import_error_placeholder("center_value")
    log_exp2cpm = _import_error_placeholder("log_exp2cpm")
    non_log2log_cpm = _import_error_placeholder("non_log2log_cpm")
    non_log2cpm = _import_error_placeholder("non_log2cpm")
    ciber_exp = _import_error_placeholder("ciber_exp")
    read_data_from_h5ad = _import_error_placeholder("read_data_from_h5ad")
    create_h5ad_dataset = _import_error_placeholder("create_h5ad_dataset")
    read_df = _import_error_placeholder("read_df")
    calculate_rmse = _import_error_placeholder("calculate_rmse")
    calculate_r2 = _import_error_placeholder("calculate_r2")
    calculate_mae = _import_error_placeholder("calculate_mae")
    cal_relative_error = _import_error_placeholder("cal_relative_error")
    check_dir = _import_error_placeholder("check_dir")
    print_df = _import_error_placeholder("print_df")
    print_msg = _import_error_placeholder("print_msg")
    default_core_marker_genes = _import_error_placeholder("default_core_marker_genes")
    read_marker_gene = _import_error_placeholder("read_marker_gene")
    sorted_cell_types = _import_error_placeholder("sorted_cell_types")
    get_inx2cell_type = _import_error_placeholder("get_inx2cell_type")
    do_pca_analysis = _import_error_placeholder("do_pca_analysis")
    do_umap_analysis = _import_error_placeholder("do_umap_analysis")
    get_corr = _import_error_placeholder("get_corr")
    get_corr_spearman = _import_error_placeholder("get_corr_spearman")

# Key plotting functions
try:
    from .plot import (
        plot_loss,
        compare_y_y_pred_plot,
        plot_gene_exp,
        plot_marker_gene_in_cell_type,
        plot_pca,
        plot_clustermap,
        ScatterPlot,
    )
except ImportError:
    plot_loss = _import_error_placeholder("plot_loss")
    compare_y_y_pred_plot = _import_error_placeholder("compare_y_y_pred_plot")
    plot_gene_exp = _import_error_placeholder("plot_gene_exp")
    plot_marker_gene_in_cell_type = _import_error_placeholder("plot_marker_gene_in_cell_type")
    plot_pca = _import_error_placeholder("plot_pca")
    plot_clustermap = _import_error_placeholder("plot_clustermap")
    ScatterPlot = _import_error_placeholder("ScatterPlot")

# Simulation functions
try:
    from .simulation import (
        BulkGEPGenerator,
        SingleCellTypeGEPGenerator,
        segment_generation_fraction,
        random_generation_fraction,
        fragment_generation_fraction,
    )
except ImportError:
    BulkGEPGenerator = _import_error_placeholder("BulkGEPGenerator")
    SingleCellTypeGEPGenerator = _import_error_placeholder("SingleCellTypeGEPGenerator")
    segment_generation_fraction = _import_error_placeholder("segment_generation_fraction")
    random_generation_fraction = _import_error_placeholder("random_generation_fraction")
    fragment_generation_fraction = _import_error_placeholder("fragment_generation_fraction")

# Workflow functions
try:
    from .workflow import run_step3, run_step4, tcga_evaluation
except ImportError:
    run_step3 = _import_error_placeholder("run_step3")
    run_step4 = _import_error_placeholder("run_step4") 
    tcga_evaluation = _import_error_placeholder("tcga_evaluation")

# Convenience functions for simplified usage
try:
    from .convenience import (
        quick_deconvolution,
        load_and_preprocess,
        evaluate_predictions,
        create_training_workflow,
    )
except ImportError:
    quick_deconvolution = _import_error_placeholder("quick_deconvolution")
    load_and_preprocess = _import_error_placeholder("load_and_preprocess")
    evaluate_predictions = _import_error_placeholder("evaluate_predictions")
    create_training_workflow = _import_error_placeholder("create_training_workflow")

# Expose all main functionality at package level
__all__ = [
    # Main class
    'DeSide',
    
    # Data reading classes
    'ReadH5AD',
    'ReadExp', 
    
    # Data preprocessing
    'filter_gene_by_expression_log_mean',
    'filter_gene_by_expression_min_max',
    'filter_sample_by_expression',
    'filter_gene_by_variance',
    'log2_transform',
    'center_value',
    
    # Data format conversion
    'log_exp2cpm',
    'non_log2log_cpm',
    'non_log2cpm', 
    'ciber_exp',
    
    # File I/O
    'read_data_from_h5ad',
    'create_h5ad_dataset',
    'read_df',
    
    # Evaluation
    'calculate_rmse',
    'calculate_r2', 
    'calculate_mae',
    'cal_relative_error',
    
    # Utilities
    'check_dir',
    'print_df',
    'print_msg',
    
    # Cell types and markers
    'default_core_marker_genes',
    'read_marker_gene',
    'sorted_cell_types',
    'get_inx2cell_type',
    
    # Analysis
    'do_pca_analysis',
    'do_umap_analysis',
    'get_corr',
    'get_corr_spearman',
    
    # Plotting
    'plot_loss',
    'compare_y_y_pred_plot',
    'plot_gene_exp',
    'plot_marker_gene_in_cell_type',
    'plot_pca',
    'plot_clustermap',
    'ScatterPlot',
    
    # Simulation
    'BulkGEPGenerator',
    'SingleCellTypeGEPGenerator', 
    'segment_generation_fraction',
    'random_generation_fraction',
    'fragment_generation_fraction',
    
    # Workflow
    'run_step3',
    'run_step4',
    'tcga_evaluation',
    
    # Convenience functions
    'quick_deconvolution',
    'load_and_preprocess',
    'evaluate_predictions',
    'create_training_workflow',
]
