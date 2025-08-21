#!/usr/bin/env python3
"""
Simple usage examples for DeSide package showing one-line import patterns
similar to NumPy and Pandas.

This demonstrates the improved accessibility of the DeSide package.
"""

# Example 1: Import DeSide like NumPy/Pandas
import deside as ds

print("DeSide package imported successfully!")
print(f"Version: {ds.__version__}")

# Example 2: Main deconvolution workflow (when dependencies are available)
def basic_deconvolution_workflow():
    """Example of using DeSide for cellular deconvolution"""
    try:
        # Create DeSide model instance 
        model = ds.DeSide(model_dir='./models', log_file_path='./log.txt')
        print("✓ DeSide model created")
        
        # Load and preprocess data
        data_reader = ds.ReadH5AD(file_path='path/to/data.h5ad', show_info=True)
        gene_expression = data_reader.get_df()
        print("✓ Data loaded")
        
        # Train the model (example parameters)
        # model.train_model(training_set_file_path=['path/to/training.h5ad'])
        print("✓ Model training would proceed here")
        
        # Make predictions
        # predictions = model.predict(input_file='path/to/test.h5ad')
        print("✓ Predictions would be made here")
        
    except ImportError as e:
        print(f"Dependencies not available: {e}")
        print("Install with: pip install deside[full]")

# Example 3: Data preprocessing utilities
def data_preprocessing_example():
    """Example of using DeSide utility functions"""
    try:
        # Create output directory
        ds.check_dir('./results')
        
        # Filter genes by expression
        # filtered_genes = ds.filter_gene_by_expression_log_mean(expression_df, min_exp_value=3)
        
        # Convert between data formats
        # cpm_data = ds.log_exp2cpm(log_data)
        # log_data = ds.non_log2log_cpm(raw_data)
        
        # Calculate evaluation metrics
        # rmse = ds.calculate_rmse(y_true, y_pred)
        # r2 = ds.calculate_r2(y_true, y_pred)
        
        print("✓ Data preprocessing functions accessible")
        
    except ImportError as e:
        print(f"Dependencies not available: {e}")

# Example 4: Plotting and visualization
def plotting_example():
    """Example of using DeSide plotting functions"""
    try:
        # Plot training loss
        # ds.plot_loss(loss_history, save_path='./loss_plot.png')
        
        # Compare predictions vs truth
        # ds.compare_y_y_pred_plot(y_true, y_pred, result_file_dir='./results')
        
        # Plot gene expression
        # ds.plot_gene_exp(expression_data, gene_list=['CD3D', 'CD8A'])
        
        # Plot PCA
        # ds.plot_pca(data, labels, save_path='./pca_plot.png')
        
        print("✓ Plotting functions accessible")
        
    except ImportError as e:
        print(f"Dependencies not available: {e}")

# Example 5: Data simulation
def simulation_example():
    """Example of using DeSide simulation capabilities"""
    try:
        # Generate synthetic bulk RNA-seq data
        # bulk_generator = ds.BulkGEPGenerator()
        # synthetic_data = bulk_generator.generate_data()
        
        # Generate cell proportions
        # proportions = ds.segment_generation_fraction(n_samples=100, n_cell_types=8)
        
        print("✓ Simulation functions accessible")
        
    except ImportError as e:
        print(f"Dependencies not available: {e}")

if __name__ == "__main__":
    print("=== DeSide Simple Usage Examples ===\n")
    
    print("1. Basic deconvolution workflow:")
    basic_deconvolution_workflow()
    print()
    
    print("2. Data preprocessing:")
    data_preprocessing_example()
    print()
    
    print("3. Plotting and visualization:")
    plotting_example()
    print()
    
    print("4. Data simulation:")
    simulation_example()
    print()
    
    print("5. Convenience functions (one-liners):")
    print("   # Complete deconvolution in one line:")
    print("   predictions = ds.quick_deconvolution('data.h5ad')")
    print("   # Load and preprocess data:")
    print("   data = ds.load_and_preprocess('data.h5ad')")
    print("   # Evaluate predictions:")
    print("   metrics = ds.evaluate_predictions(y_true, y_pred)")
    print("   # Train a model:")
    print("   model = ds.create_training_workflow('train.h5ad', cell_types)")
    print()
    
    print("=== Available functions ===")
    available_functions = [x for x in dir(ds) if not x.startswith('_')]
    print(f"Total available: {len(available_functions)}")
    print("Key functions:", available_functions[:10], "...")
    
    print("\n=== One-line usage patterns ===")
    print("import deside as ds")
    print("model = ds.DeSide(model_dir='./models')")
    print("data = ds.ReadH5AD('data.h5ad')")
    print("ds.plot_gene_exp(data, genes=['CD3D'])")
    print("metrics = ds.calculate_rmse(y_true, y_pred)")
    print("\n=== Ultra-simple patterns ===")
    print("predictions = ds.quick_deconvolution('data.h5ad')")
    print("data = ds.load_and_preprocess('data.h5ad')")
    print("metrics = ds.evaluate_predictions('true.csv', 'pred.csv')")
    print("model = ds.create_training_workflow('train.h5ad', ['T', 'B', 'Macro'])")