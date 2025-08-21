"""
Convenience functions for simplified DeSide usage
"""

def quick_deconvolution(data_file, model_dir='./deside_model', output_file='predictions.csv'):
    """
    Perform cellular deconvolution with a single function call.
    
    Parameters:
    -----------
    data_file : str
        Path to input data file (.h5ad format)
    model_dir : str
        Directory for DeSide model (default: './deside_model')
    output_file : str
        Path for output predictions (default: 'predictions.csv')
    
    Returns:
    --------
    pandas.DataFrame
        Predicted cell fractions
    
    Example:
    --------
    >>> import deside as ds
    >>> predictions = ds.quick_deconvolution('bulk_data.h5ad')
    """
    try:
        from .decon_cf import DeSide
        from .utility.read_file import ReadH5AD
        
        # Create model
        model = DeSide(model_dir=model_dir)
        
        # Make predictions
        predictions = model.predict(input_file=data_file, output_file_path=output_file)
        
        return predictions
        
    except ImportError as e:
        raise ImportError(f"Dependencies not available for quick_deconvolution: {e}")


def load_and_preprocess(data_file, filter_genes=True, min_exp_value=3, max_exp_value=10):
    """
    Load and preprocess gene expression data with default settings.
    
    Parameters:
    -----------
    data_file : str
        Path to input data file (.h5ad format)
    filter_genes : bool
        Whether to filter genes by expression (default: True)
    min_exp_value : float
        Minimum expression value for gene filtering (default: 3)
    max_exp_value : float
        Maximum expression value for gene filtering (default: 10)
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed gene expression matrix
    
    Example:
    --------
    >>> import deside as ds
    >>> expression_data = ds.load_and_preprocess('data.h5ad')
    """
    try:
        from .utility.read_file import ReadH5AD
        from .utility import filter_gene_by_expression_log_mean
        
        # Load data
        data_reader = ReadH5AD(data_file, show_info=True)
        expression_data = data_reader.get_df()
        
        # Filter genes if requested
        if filter_genes:
            expression_data = filter_gene_by_expression_log_mean(
                expression_data, 
                min_exp_value=min_exp_value, 
                max_exp_value=max_exp_value
            )
        
        return expression_data
        
    except ImportError as e:
        raise ImportError(f"Dependencies not available for load_and_preprocess: {e}")


def evaluate_predictions(y_true, y_pred, plot_results=True, output_dir='./results'):
    """
    Evaluate deconvolution predictions with common metrics and plots.
    
    Parameters:
    -----------
    y_true : pandas.DataFrame or str
        True cell fractions or path to file containing them
    y_pred : pandas.DataFrame or str  
        Predicted cell fractions or path to file containing them
    plot_results : bool
        Whether to generate comparison plots (default: True)
    output_dir : str
        Directory for output plots (default: './results')
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    
    Example:
    --------
    >>> import deside as ds
    >>> metrics = ds.evaluate_predictions('true_fractions.csv', 'predictions.csv')
    """
    try:
        from .utility import calculate_rmse, calculate_r2, calculate_mae, check_dir
        from .plot import compare_y_y_pred_plot
        import pandas as pd
        
        # Load data if paths provided
        if isinstance(y_true, str):
            y_true = pd.read_csv(y_true, index_col=0)
        if isinstance(y_pred, str):
            y_pred = pd.read_csv(y_pred, index_col=0)
        
        # Calculate metrics
        metrics = {
            'RMSE': calculate_rmse(y_true, y_pred),
            'R2': calculate_r2(y_true, y_pred),
            'MAE': calculate_mae(y_true, y_pred)
        }
        
        # Generate plots if requested
        if plot_results:
            check_dir(output_dir)
            compare_y_y_pred_plot(y_true, y_pred, result_file_dir=output_dir)
        
        return metrics
        
    except ImportError as e:
        raise ImportError(f"Dependencies not available for evaluate_predictions: {e}")


def create_training_workflow(training_data, cell_types, model_dir='./deside_model', 
                           log_file='./training.log'):
    """
    Create and train a DeSide model with simplified interface.
    
    Parameters:
    -----------
    training_data : str or list
        Path to training data file(s) (.h5ad format)
    cell_types : list
        List of cell types to predict
    model_dir : str
        Directory to save the trained model (default: './deside_model')
    log_file : str
        Path for training log file (default: './training.log')
    
    Returns:
    --------
    DeSide
        Trained DeSide model instance
    
    Example:
    --------
    >>> import deside as ds
    >>> cell_types = ['T cells', 'B cells', 'Macrophages']
    >>> model = ds.create_training_workflow('training.h5ad', cell_types)
    """
    try:
        from .decon_cf import DeSide
        from .utility import check_dir
        
        # Ensure training_data is a list
        if isinstance(training_data, str):
            training_data = [training_data]
        
        # Create directories
        check_dir(model_dir)
        
        # Create and train model
        model = DeSide(model_dir=model_dir, log_file_path=log_file)
        
        # Default hyperparameters for quick start
        default_params = {
            'architecture': ([200, 2000, 2000, 2000, 50], [0.05, 0.05, 0.05, 0.2, 0]),
            'loss_function_alpha': 0.5,
            'normalization': 'layer_normalization',
            'normalization_layer': [0, 0, 1, 1, 1, 1],
            'pathway_network': True,
            'last_layer_activation': 'sigmoid',
            'learning_rate': 1e-4,
            'batch_size': 128
        }
        
        # Train the model
        model.train_model(
            training_set_file_path=training_data,
            hyper_params=default_params,
            cell_types=cell_types,
            scaling_by_constant=True,
            scaling_by_sample=False,
            n_patience=100,
            n_epoch=3000,
            verbose=1
        )
        
        return model
        
    except ImportError as e:
        raise ImportError(f"Dependencies not available for create_training_workflow: {e}")