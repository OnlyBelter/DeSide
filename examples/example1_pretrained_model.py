import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import deside


def _parse_args():
    parser = argparse.ArgumentParser(
        prog='example1_pretrained_model',
        description='Run DeSide Example 1 (pre-trained model) from the command line.',
    )
    parser.add_argument('--input-file', required=True, help='Bulk GEP file path (TPM by default).')
    parser.add_argument('--output-file', required=True, help='Output CSV path for predicted cell fractions.')
    parser.add_argument('--model-dir', default='./DeSide_model', help='Directory containing pre-trained model files.')
    parser.add_argument('--dataset-dir', default='./datasets', help='Directory containing datasets/gene_set/.')
    parser.add_argument('--exp-type', default='TPM', choices=['TPM', 'log_space'], help='Input expression type.')
    parser.add_argument(
        '--transpose',
        default='true',
        choices=['true', 'false'],
        help='Set to true if input is genes by samples (the default in the notebook).',
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    transpose = args.transpose.lower() == 'true'
    deside.predict_with_pretrained_model(
        input_file=args.input_file,
        output_file_path=args.output_file,
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        exp_type=args.exp_type,
        transpose=transpose,
    )


if __name__ == '__main__':
    main()
