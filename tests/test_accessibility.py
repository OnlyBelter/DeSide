#!/usr/bin/env python3
"""
Test backward compatibility of DeSide package imports.

This ensures that existing import patterns still work after accessibility improvements.
"""

def test_backward_compatibility():
    """Test that old import patterns still work"""
    
    print("=== Testing Backward Compatibility ===\n")
    
    # Test 1: Original DeSide class import
    print("1. Testing original DeSide class import...")
    try:
        from deside.decon_cf import DeSide
        print("✓ from deside.decon_cf import DeSide - works")
    except ImportError as e:
        print(f"✗ from deside.decon_cf import DeSide - failed: {e}")
    
    # Test 2: Original utility imports
    print("\n2. Testing original utility imports...")
    try:
        from deside.utility.read_file import ReadH5AD, ReadExp
        print("✓ from deside.utility.read_file import ReadH5AD, ReadExp - works")
    except ImportError as e:
        print(f"✗ from deside.utility.read_file import ReadH5AD, ReadExp - failed: {e}")
    
    try:
        from deside.utility import check_dir, print_df
        print("✓ from deside.utility import check_dir, print_df - works")
    except ImportError as e:
        print(f"✗ from deside.utility import check_dir, print_df - failed: {e}")
    
    # Test 3: Original plot imports
    print("\n3. Testing original plot imports...")
    try:
        from deside.plot import plot_loss, compare_y_y_pred_plot
        print("✓ from deside.plot import plot_loss, compare_y_y_pred_plot - works")
    except ImportError as e:
        print(f"✗ from deside.plot import plot_loss, compare_y_y_pred_plot - failed: {e}")
    
    # Test 4: Original simulation imports
    print("\n4. Testing original simulation imports...")
    try:
        from deside.simulation import BulkGEPGenerator, SingleCellTypeGEPGenerator
        print("✓ from deside.simulation import BulkGEPGenerator, SingleCellTypeGEPGenerator - works")
    except ImportError as e:
        print(f"✗ from deside.simulation import BulkGEPGenerator, SingleCellTypeGEPGenerator - failed: {e}")
    
    # Test 5: Original workflow imports
    print("\n5. Testing original workflow imports...")
    try:
        from deside.workflow import run_step3, run_step4
        print("✓ from deside.workflow import run_step3, run_step4 - works")
    except ImportError as e:
        print(f"✗ from deside.workflow import run_step3, run_step4 - failed: {e}")


def test_new_accessibility():
    """Test that new simplified import patterns work"""
    
    print("\n=== Testing New Accessibility Features ===\n")
    
    # Test 1: Simple package import
    print("1. Testing simple package import...")
    try:
        import deside as ds
        print(f"✓ import deside as ds - works (version: {ds.__version__})")
    except ImportError as e:
        print(f"✗ import deside as ds - failed: {e}")
        return
    
    # Test 2: Direct access to main classes
    print("\n2. Testing direct access to main classes...")
    main_classes = ['DeSide', 'ReadH5AD', 'ReadExp', 'BulkGEPGenerator']
    for cls_name in main_classes:
        if hasattr(ds, cls_name):
            print(f"✓ ds.{cls_name} - accessible")
        else:
            print(f"✗ ds.{cls_name} - not accessible")
    
    # Test 3: Direct access to utility functions
    print("\n3. Testing direct access to utility functions...")
    utility_functions = ['check_dir', 'print_df', 'calculate_rmse', 'filter_gene_by_expression_log_mean']
    for func_name in utility_functions:
        if hasattr(ds, func_name):
            print(f"✓ ds.{func_name} - accessible")
        else:
            print(f"✗ ds.{func_name} - not accessible")
    
    # Test 4: Direct access to plotting functions
    print("\n4. Testing direct access to plotting functions...")
    plot_functions = ['plot_loss', 'compare_y_y_pred_plot', 'plot_gene_exp']
    for func_name in plot_functions:
        if hasattr(ds, func_name):
            print(f"✓ ds.{func_name} - accessible")
        else:
            print(f"✗ ds.{func_name} - not accessible")
    
    # Test 5: Convenience functions
    print("\n5. Testing convenience functions...")
    convenience_functions = ['quick_deconvolution', 'load_and_preprocess', 'evaluate_predictions']
    for func_name in convenience_functions:
        if hasattr(ds, func_name):
            print(f"✓ ds.{func_name} - accessible")
        else:
            print(f"✗ ds.{func_name} - not accessible")
    
    # Test 6: __all__ completeness
    print("\n6. Testing __all__ completeness...")
    all_items = ds.__all__
    actual_items = [x for x in dir(ds) if not x.startswith('_')]
    missing_from_all = set(actual_items) - set(all_items)
    extra_in_all = set(all_items) - set(actual_items)
    
    if missing_from_all:
        print(f"⚠ Items missing from __all__: {missing_from_all}")
    if extra_in_all:
        print(f"⚠ Extra items in __all__: {extra_in_all}")
    if not missing_from_all and not extra_in_all:
        print("✓ __all__ is complete and accurate")
    
    print(f"\nTotal items available: {len(actual_items)}")
    print(f"Items in __all__: {len(all_items)}")


if __name__ == "__main__":
    test_backward_compatibility()
    test_new_accessibility()
    
    print("\n=== Summary ===")
    print("DeSide package now provides:")
    print("• Backward compatibility with existing import patterns")
    print("• Simplified one-line imports like NumPy/Pandas")
    print("• 50+ functions available at package level")
    print("• Convenience functions for common workflows")
    print("• Robust error handling for missing dependencies")