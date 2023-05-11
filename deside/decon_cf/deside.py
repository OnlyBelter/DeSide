import os
import json
import functools
import numpy as np
import pandas as pd
from typing import Union
import tensorflow as tf
from tensorflow import keras
from ..utility.read_file import ReadH5AD, ReadExp
from ..utility import check_dir, print_msg
from ..plot import plot_loss


class DeSide(object):
    def __init__(self, model_dir: str, log_file_path: str = None, model_name: str = 'DeSide'):
        """
        :param model_dir: the directory of saving well-trained model
        :param log_file_path: the file path of log
        :param model_name: only for naming
        """
        self.model_dir = model_dir
        self.model = None
        self.cell_types = None
        self.gene_list = None
        self.model_name = model_name
        self.min_cell_fraction = 0.0001  # set to 0 if less than this value in predicted cell fractions
        self.model_file_path = os.path.join(self.model_dir, f'model_{model_name}.h5')
        self.cell_type_file_path = os.path.join(self.model_dir, 'celltypes.txt')
        self.gene_list_file_path = os.path.join(self.model_dir, 'genes.txt')
        self.training_set_file_path = None
        self.hyper_params = None
        self.one_minus_alpha = False
        if log_file_path is None:
            log_file_path = os.path.join(self.model_dir, 'log.txt')
        self.log_file_path = log_file_path
        check_dir(self.model_dir)

    def build_model(self, input_shape, output_shape, hyper_params):
        """
        :param input_shape: the number of features (genes)
        :param output_shape: the dimension of output (number of cell types to predict cell fraction)
        :param hyper_params: pre-determined hyper-parameters for DeSide model
        """
        self.hyper_params = hyper_params
        hidden_units = hyper_params['architecture'][0]
        dropout_rates = hyper_params['architecture'][1]
        using_batch_normalization = hyper_params['batch_normalization']
        last_layer_activation_function = hyper_params['last_layer_activation']
        if last_layer_activation_function == 'hard_sigmoid':
            last_layer_activation_function = keras.activations.hard_sigmoid

        # remove bias when using BatchNormalization
        dense = functools.partial(keras.layers.Dense, use_bias=False, kernel_initializer='he_normal')
        batch_normalization = keras.layers.BatchNormalization
        activation = functools.partial(keras.layers.Activation, activation='relu')  # activate after BatchNormalization

        if using_batch_normalization:
            gep = keras.Input(shape=(input_shape,), name='input')
            gep_normalized = batch_normalization()(gep)
            features = dense(units=hidden_units[0])(gep_normalized)  # the first dense layer
            features = batch_normalization()(features)
            features = activation()(features)
            for n_units in hidden_units[1:-1]:
                features = dense(units=n_units)(features)
                features = batch_normalization()(features)
                features = activation()(features)
            features = dense(units=hidden_units[-1], use_bias=True, activation='relu')(features)  # before output
            y_pred = dense(units=output_shape, use_bias=True, activation=last_layer_activation_function)(features)
            model = keras.Model(inputs=gep, outputs=y_pred, name='DeSide')
        else:
            gep = keras.Input(shape=(input_shape,), name='input')
            features = dense(units=hidden_units[0], use_bias=True, activation='relu')(gep)  # the first dense layer
            if dropout_rates[0] > 0:
                features = keras.layers.Dropout(dropout_rates[0])(features)
            hid_dropout = list(zip(hidden_units[1:], dropout_rates[1:]))
            for n_units, dropout_rate in hid_dropout:
                features = dense(units=n_units, use_bias=True, activation='relu')(features)
                if dropout_rate > 0:
                    features = keras.layers.Dropout(dropout_rate)(features)
            y_pred = dense(units=output_shape, use_bias=True, activation=last_layer_activation_function)(features)
            model = keras.Model(inputs=gep, outputs=y_pred, name='DeSide')
        self.model = model

    def train_model(self, training_set_file_path: Union[str, list], hyper_params: dict,
                    cell_types: list = None, scaling_by_sample: bool = True, callback: bool = True,
                    n_epoch: int = 10000, metrics: str = 'mse', n_patience: int = 100, scaling_by_constant=False,
                    remove_cancer_cell=False, fine_tune=False, one_minus_alpha: bool = False, verbose=1):
        """
        :param training_set_file_path: the file path of training set, .h5ad file, log2cpm1p format, samples by genes
        :param hyper_params: pre-determined hyper-parameters for DeSide model
        :param cell_types: specific a list of cell types instead of using all cell types in training set
        :param scaling_by_sample: whether to scale the expression values of each sample to [0, 1] by 'min_max'
        :param callback: whether to use callback function when training model
        :param n_epoch: the max number of epochs to train
        :param metrics: mse (regression model) / accuracy (classifier)
        :param n_patience: patience in early_stopping_callback
        :param remove_cancer_cell: remove cancer cell from y if True, using "1-others"
        :param fine_tune: fine tune pre-trained model
        :param scaling_by_constant:
        :param one_minus_alpha: use 1 - alpha for all cell types if True
        :param verbose: 0: silent, 1: progress bar, 2: one line per epoch
        """
        self.one_minus_alpha = one_minus_alpha
        if not os.path.exists(self.model_file_path):
            print_msg('Start to training model...', log_file_path=self.log_file_path)
            learning_rate = hyper_params['learning_rate']
            loss_function = hyper_params['loss_function']
            batch_size = hyper_params['batch_size']

            # read training set
            if type(training_set_file_path) == str:
                training_set_file_path = [training_set_file_path]
            x_list, y_list = [], []
            print_msg('Start to reading training set...', log_file_path=self.log_file_path)
            for file_path in training_set_file_path:
                file_obj = ReadH5AD(file_path)
                _x = file_obj.get_df()  # bulk cell GEPs, samples by genes
                print('x shape:', _x.shape, file_path)
                print('x head:', _x.head())
                _y = file_obj.get_cell_fraction()  # cell fractions

                x_list.append(_x.copy())
                y_list.append(_y.copy())
            x = pd.concat(x_list, join='inner', axis=0)
            y = pd.concat(y_list)
            if self.one_minus_alpha:
                y = 1 - y
            del file_obj, _x, _y, x_list, y_list

            # scaling x
            x_obj = ReadExp(x, exp_type='log_space')
            if len(training_set_file_path) >= 2:
                x_obj.to_tpm()  # if multiple training set, rescale to TPM, in case some genes were removed
                x_obj.to_log2cpm1p()
            if scaling_by_sample:
                x_obj.do_scaling()
            if scaling_by_constant:
                # file_obj = ReadExp(_x, exp_type='log_space')
                x_obj.do_scaling_by_constant()
            x = x_obj.get_exp()

            self.gene_list = x.columns.to_list()  # a list of all gene names

            # filtering cell types in cell fraction file, for example removing the cell fraction of cancer cell
            if cell_types is None:
                self.cell_types = y.columns.to_list()  # a list
            else:
                self.cell_types = cell_types
            if remove_cancer_cell:
                self.cell_types = [i for i in self.cell_types if i != 'Cancer Cells']
            y = y.loc[:, y.columns.isin(self.cell_types)]

            # Save features and cell types
            pd.DataFrame(self.cell_types).to_csv(self.cell_type_file_path, sep="\t")
            pd.DataFrame(self.gene_list).to_csv(self.gene_list_file_path, sep="\t")

            print(f'   Using cell types: {self.cell_types}')
            print(f'   The shape of X is: {x.shape}, (n_sample, n_gene)')
            print(f'   The shape of y is: {y.shape}, (n_sample, n_cell_type)')
            if not fine_tune:
                self.build_model(input_shape=len(self.gene_list), output_shape=len(self.cell_types),
                                 hyper_params=hyper_params)
            if self.model is None:
                raise FileNotFoundError('pre-trained model should be assigned to self.model')
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
            _loss_function = loss_function
            _metrics = [metrics]
            if loss_function == 'mae+rmse':
                _loss_function = loss_fn_mae_rmse
                _metrics = ['mae', keras.metrics.RootMeanSquaredError()]
            self.model.compile(optimizer=opt, loss=_loss_function, metrics=_metrics)
            print(self.model.summary())

            # training model
            if callback:
                # Stop training when a monitored metric has stopped improving.
                # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
                early_stopping_callback = keras.callbacks.EarlyStopping(
                    # patience=10,
                    patience=n_patience,
                    monitor='val_loss',
                    mode='min',
                    restore_best_weights=True)
                history = self.model.fit(x.values, y.values, epochs=n_epoch,
                                         batch_size=batch_size, verbose=verbose, validation_split=0.2,
                                         callbacks=[early_stopping_callback])
                # history = self.model.fit(x.values, y.values, epochs=n_epoch,
                #                          batch_size=batch_size, verbose=2)

            else:
                history = self.model.fit(x.values, y.values, epochs=n_epoch,
                                         batch_size=batch_size, verbose=verbose, validation_split=0.2)

            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            hist.to_csv(os.path.join(self.model_dir, 'history_reg.csv'))

            self.model.save(self.model_file_path)
            plot_loss(hist, output_dir=self.model_dir, y_label=loss_function)
            key_params_file_path = os.path.join(self.model_dir, 'key_params.txt')
            print(f'   Key parameters during model training will be saved in {key_params_file_path}.')
            self.save_params(key_params_file_path)
            print_msg('Training done.', log_file_path=self.log_file_path)
        else:
            print(f'Previous model existed: {self.model_file_path}')

    def get_x_before_predict(self, input_file, exp_type, transpose: bool = False, print_info: bool = True,
                             scaling_by_sample: bool = False, scaling_by_constant: bool = True):
        """
        :param input_file: input file path
        :param exp_type: 'log_space' or 'raw_space'
        :param transpose: if True, transpose the input dataframe
        :param print_info: if True, print info
        :param scaling_by_sample: if True, scaling by sample
        :param scaling_by_constant: if True, scaling by constant
        :return: x
        """
        if self.gene_list is None:
            self.gene_list = self.get_gene_list()
        if exp_type not in ['TPM', 'log_space']:
            raise ValueError(f'exp_type should be "TPM" or "log_space", "{exp_type}" is invalid.')
        if '.h5ad' in input_file:
            read_h5ad_obj = ReadH5AD(input_file)
            _input_data = read_h5ad_obj.get_df()  # df, samples by genes
            read_df_obj = ReadExp(_input_data, exp_type=exp_type, transpose=transpose)
        elif np.any([i in input_file for i in ['.csv', '.txt', '.tsv']]) or (type(input_file) == pd.DataFrame):
            read_df_obj = ReadExp(input_file, exp_type=exp_type, transpose=transpose)
        else:
            raise Exception(f'The current file path of raw data is {input_file}, '
                            f'only "*_.csv", "*_.txt", "*_.tsv", or "*_.h5ad" is supported, ' 
                            f'please check the file path and try again.')
        # check gene list
        read_df_obj.align_with_gene_list(gene_list=self.gene_list, fill_not_exist=True)
        if exp_type != 'log_space':
            read_df_obj.to_log2cpm1p()
        if scaling_by_sample:
            read_df_obj.do_scaling()
        if scaling_by_constant:
            read_df_obj.do_scaling_by_constant()
        x = read_df_obj.get_exp()  # a DataFrame with log2(TPM + 1) gene expression values, samples by genes
        # make sure that only genes in self.gene_list were used and keep the same order
        # check duplication of gene names
        _gene_list = x.columns.to_list()
        if not (len(_gene_list) == len(set(_gene_list))):
            Warning(
                "Current input file contains duplicate genes! The first occurring gene will be kept.")
            x = x.loc[:, ~x.columns.duplicated(keep='first')]
        assert np.all(x.columns == self.gene_list), 'The gene list in input file is not the same as the ' \
                                                    'gene list in pre-trained model.'
        if print_info:
            print(f'   > {len(self.gene_list)} genes included in pre-trained model and will be used for prediction.')
            print(f'   The shape of X is: {x.shape}, (n_sample, n_gene)')
        return x

    def predict(self, input_file, exp_type, output_file_path: str = None,
                transpose: bool = False, print_info: bool = True, add_cell_type: bool = False,
                scaling_by_constant=False, scaling_by_sample=True, one_minus_alpha: bool = False):
        """
        :param input_file: the file path of input file (.csv / .h5ad / pd.Dataframe), samples x genes
            simulated (or TCGA) bulk expression profiles, log2(TPM + 1) or TPM
        :param output_file_path: the file path to save prediction result
        :param exp_type: log_space or TPM, log_space means log2(TPM + 1)
        :param transpose: transpose if exp_file formed as genes (index) by samples (columns)
        :param print_info: print information during prediction
        :param add_cell_type: only True when predicting cell types using classification model
        :param scaling_by_constant: scaling log2(TPM + 1) by dividing 20
        :param scaling_by_sample: scaling by sample, same as Scaden
        :param one_minus_alpha: use 1 - alpha for all cell types if True
        """
        self.one_minus_alpha = one_minus_alpha
        if print_info:
            print('   Start to predict cell fractions by pre-trained model...')
        if self.cell_types is None:
            self.cell_types = self.get_cell_type()

        # load input data
        x = self.get_x_before_predict(input_file, exp_type, transpose=transpose, print_info=print_info,
                                      scaling_by_constant=scaling_by_constant, scaling_by_sample=scaling_by_sample)

        # load pre-trained model
        if self.model is None:
            try:
                self.model = keras.models.load_model(self.model_file_path)
            except ValueError:
                self.model = keras.models.load_model(self.model_file_path,
                                                     custom_objects={'loss_fn_mae_rmse': loss_fn_mae_rmse})
            finally:
                print(f'   Pre-trained model loaded from {self.model_file_path}.')
        # predict using loaded model
        pred_result = self.model.predict(x)
        pred_df = pd.DataFrame(pred_result, index=x.index, columns=self.cell_types)
        if self.one_minus_alpha:
            pred_df = 1 - pred_df
        pred_df[pred_df.values < self.min_cell_fraction] = 0
        # pred_df.to_csv(out_name, sep="\t")
        # rescaling to 1 if the sum of all cell types > 1
        for sample_id, row in pred_df.iterrows():
            if np.sum(row) > 1:
                pred_df.loc[sample_id] = row / np.sum(row)

        # Calculate 1-others
        if 'Cancer Cells' not in pred_df.columns:
            pred_df_with_1_others = pred_df.loc[:, [i for i in pred_df.columns if i != 'Cancer Cells']].copy()
            pred_df_with_1_others['1-others'] = 1 - np.vstack(pred_df_with_1_others.sum(axis=1))
            pred_df_with_1_others.loc[pred_df_with_1_others['1-others'] < 0, '1-others'] = 0
            pred_df_with_1_others['Cancer Cells'] = pred_df_with_1_others['1-others']
            pred_df = pred_df_with_1_others.copy()
        # pred_df_with_1_others.to_csv(output_file_path, float_format='%.3f')
        if add_cell_type:
            pred_df['pred_cell_type'] = self.pred_cell_type_by_cell_frac(pred_cell_frac=pred_df)
        if print_info:
            print('   Model prediction done.')
        if output_file_path is not None:
            pred_df.to_csv(output_file_path, float_format='%.3f')
        else:
            return pred_df

    def get_model(self):
        if (self.model is None) and (os.path.exists(self.model_file_path)):
            try:
                self.model = keras.models.load_model(self.model_file_path)
            except ValueError:
                self.model = keras.models.load_model(self.model_file_path,
                                                     custom_objects={'loss_fn_mae_rmse': loss_fn_mae_rmse})
            finally:
                print(f'   Pre-trained model loaded from {self.model_file_path}.')

        return self.model

    def get_parameters(self) -> dict:
        key_params = {
            'model_name': self.model_name, 'model_file_path': self.model_file_path,
            'hyper_params': self.hyper_params, 'training_set_file_path': self.training_set_file_path,
            'cell_type_file_path': self.cell_type_file_path,
            'gene_list_file_path': self.gene_list_file_path, 'log_file_path': self.log_file_path
        }
        return key_params

    def get_gene_list(self) -> list:
        if (self.gene_list is None) and os.path.exists(self.gene_list_file_path):
            self.gene_list = list(pd.read_csv(self.gene_list_file_path, sep='\t', index_col=0)['0'])
        return self.gene_list

    def get_cell_type(self) -> list:
        if (self.cell_types is None) and os.path.exists(self.cell_type_file_path):
            self.cell_types = list(pd.read_csv(self.cell_type_file_path, sep='\t', index_col=0)['0'])
        return self.cell_types

    def save_params(self, output_file_path: str):
        key_params = self.get_parameters()
        with open(output_file_path, 'w') as f_handle:
            json.dump(key_params, fp=f_handle, indent=2)

    def pred_cell_type_by_cell_frac(self, pred_cell_frac: pd.DataFrame) -> list:
        """
        convert predicted cell fractions to cell types
        """
        id2cell_type = {i: self.cell_types[i] for i in range(len(self.cell_types))}
        pred_id = pred_cell_frac.values.argmax(axis=1)
        return [id2cell_type[i] for i in pred_id]


def loss_fn_mae_rmse(y_true, y_pred, alpha=0.8):
    mae = keras.losses.MeanAbsoluteError()
    mse = keras.losses.MeanSquaredError()
    return alpha * mae(y_true, y_pred) + (1 - alpha) * tf.sqrt(mse(y_true, y_pred))
