import os
import json
import functools
import numpy as np
import pandas as pd
from typing import Union
import tensorflow as tf
from tensorflow import keras
from ..utility.read_file import ReadH5AD, ReadExp
from ..utility import check_dir, print_msg, get_x_by_pathway_network
from ..plot import plot_loss


class DeSide(object):
    """
    DeSide model for predicting cell proportions in bulk RNA-seq data

    :param model_dir: the directory of saving well-trained model
    :param log_file_path: the file path of log
    :param model_name: only for naming
    """

    def __init__(self, model_dir: str, log_file_path: str = None, model_name: str = 'DeSide'):
        """
        """
        self.model_dir = model_dir
        self.model = None
        self.cell_types = None
        self.gene_list = None
        self.model_name = model_name
        self.min_cell_fraction = 0.0001  # set to 0 if less than this value in predicted cell fractions
        self.model_file_path = os.path.join(self.model_dir, f'model_{model_name}.h5')
        self.cell_type_file_path = os.path.join(self.model_dir, 'celltypes.txt')
        self.gene_list_file_path = os.path.join(self.model_dir, 'genes.txt')  # the gene list used as input of model
        self.gene_list_for_gep_file_path = os.path.join(self.model_dir, 'genes_for_gep.txt')
        # the gene list used to construct pathway profiles with pathway mask (not the genes in pathway mask)
        self.gene_list_for_pathway_profile_file_path = os.path.join(self.model_dir, 'genes_for_pathway_profile.txt')
        self.training_set_file_path = None
        self.hyper_params = None
        self.one_minus_alpha = False
        if log_file_path is None:
            log_file_path = os.path.join(self.model_dir, 'log.txt')
        self.log_file_path = log_file_path
        check_dir(self.model_dir)

    def _build_model(self, input_shape, output_shape, hyper_params, n_pathway: int = 0):
        """
        :param input_shape: the number of features (genes)
        :param output_shape: the dimension of output (number of cell types to predict cell fraction)
        :param hyper_params: pre-determined hyper-parameters for DeSide model
        :param n_pathway: the number of pathways
        """
        self.hyper_params = hyper_params
        hidden_units = hyper_params['architecture'][0]
        dropout_rates = hyper_params['architecture'][1]
        normalization = hyper_params['normalization']
        normalization_layer = hyper_params.get('normalization_layer', [1] * (len(hidden_units) + 1))
        last_layer_activation_function = hyper_params['last_layer_activation']
        pathway_network = hyper_params['pathway_network']
        if last_layer_activation_function == 'hard_sigmoid':
            last_layer_activation_function = keras.activations.hard_sigmoid

        # remove bias when using BatchNormalization
        dense = functools.partial(keras.layers.Dense, use_bias=False, kernel_initializer='he_normal')
        if normalization == 'batch_normalization':
            normalization_func = keras.layers.BatchNormalization
        elif normalization == 'layer_normalization':
            normalization_func = keras.layers.LayerNormalization
        else:
            normalization_func = None
        activation = functools.partial(keras.layers.Activation, activation='relu')  # activate after BatchNormalization

        p_features = None
        pathway_profile = None
        if normalization is not None and normalization_func is not None:
            gep = keras.Input(shape=(input_shape,), name='gep')
            if normalization_layer[0] == 1:
                gep = normalization_func()(gep)
            if normalization_layer[1] == 1:
                features = dense(units=hidden_units[0])(gep)  # the first dense layer
                features = normalization_func()(features)
                features = activation()(features)
            else:
                features = dense(units=hidden_units[0], use_bias=True, activation='relu')(gep)  # the first dense layer
            if dropout_rates[0] > 0:
                features = keras.layers.Dropout(dropout_rates[0])(features)
            hu_dr_nl = list(zip(hidden_units[1:], dropout_rates[1:], normalization_layer[2:]))
            for n_units, dropout_rate, norm_layer in hu_dr_nl:
                if norm_layer == 1:
                    features = dense(units=n_units)(features)
                    features = normalization_func()(features)
                    features = activation()(features)
                else:
                    features = dense(units=n_units, use_bias=True, activation='relu')(features)
                if dropout_rate > 0:
                    features = keras.layers.Dropout(dropout_rate)(features)
            if pathway_network:
                assert 'architecture_for_pathway_network' in hyper_params, \
                    'architecture_for_pathway_network is required when using pathway network.'
                p_hidden_units = hyper_params['architecture_for_pathway_network'][0]
                p_dropout_rates = hyper_params['architecture_for_pathway_network'][1]
                pathway_profile = keras.Input(shape=(n_pathway,), name='pathway_profile')
                if normalization_layer[0] == 1:
                    pathway_profile = normalization_func()(pathway_profile)
                if normalization_layer[1] == 1:
                    p_features = dense(units=p_hidden_units[0])(pathway_profile)
                    p_features = normalization_func()(p_features)
                    p_features = activation()(p_features)
                else:
                    p_features = dense(units=p_hidden_units[0], use_bias=True, activation='relu')(pathway_profile)
                if p_dropout_rates[0] > 0:
                    p_features = keras.layers.Dropout(p_dropout_rates[0])(p_features)
                p_hu_dr_nl = list(zip(p_hidden_units[1:], p_dropout_rates[1:], normalization_layer[2:]))
                for n_units, dropout_rate, norm_layer in p_hu_dr_nl:
                    if norm_layer == 1:
                        p_features = dense(units=n_units)(p_features)
                        p_features = normalization_func()(p_features)
                        p_features = activation()(p_features)
                    else:
                        p_features = dense(units=n_units, use_bias=True, activation='relu')(p_features)
                    if dropout_rate > 0:
                        p_features = keras.layers.Dropout(dropout_rate)(p_features)
        else:
            gep = keras.Input(shape=(input_shape,), name='gep')
            features = dense(units=hidden_units[0], use_bias=True, activation='relu')(gep)  # the first dense layer
            if dropout_rates[0] > 0:
                features = keras.layers.Dropout(dropout_rates[0])(features)
            hid_dropout = list(zip(hidden_units[1:], dropout_rates[1:]))
            for n_units, dropout_rate in hid_dropout:
                features = dense(units=n_units, use_bias=True, activation='relu')(features)
                if dropout_rate > 0:
                    features = keras.layers.Dropout(dropout_rate)(features)
            if pathway_network:
                assert 'architecture_for_pathway_network' in hyper_params, \
                    'architecture_for_pathway_network is required when using pathway network.'
                p_hidden_units = hyper_params['architecture_for_pathway_network'][0]
                p_dropout_rates = hyper_params['architecture_for_pathway_network'][1]
                pathway_profile = keras.Input(shape=(n_pathway,), name='pathway_profile')
                p_features = dense(units=p_hidden_units[0], use_bias=True, activation='relu')(pathway_profile)
                if p_dropout_rates[0] > 0:
                    p_features = keras.layers.Dropout(p_dropout_rates[0])(p_features)
                p_hid_dropout = list(zip(p_hidden_units[1:], p_dropout_rates[1:]))
                for n_units, dropout_rate in p_hid_dropout:
                    p_features = dense(units=n_units, use_bias=True, activation='relu')(p_features)
                    if dropout_rate > 0:
                        p_features = keras.layers.Dropout(dropout_rate)(p_features)
        if pathway_profile is not None and p_features is not None:
            # Merge all available features into a single large vector via concatenation
            x = keras.layers.concatenate([features, p_features])
            x = dense(units=hidden_units[-1], use_bias=True, activation='relu')(x)
            y_pred = dense(units=output_shape, use_bias=True, activation=last_layer_activation_function)(x)
            model = keras.Model(inputs=[gep, pathway_profile], outputs=y_pred, name='DeSide')
        else:
            y_pred = dense(units=output_shape, use_bias=True, activation=last_layer_activation_function)(features)
            model = keras.Model(inputs=gep, outputs=y_pred, name='DeSide')
        self.model = model

    def train_model(self, training_set_file_path: Union[str, list], hyper_params: dict,
                    cell_types: list = None, scaling_by_sample: bool = True, callback: bool = True,
                    n_epoch: int = 10000, metrics: str = 'mse', n_patience: int = 100, scaling_by_constant=False,
                    remove_cancer_cell=False, fine_tune=False, one_minus_alpha: bool = False, verbose=1,
                    pathway_mask=None, method_adding_pathway='add_to_end', input_gene_list: str = None,
                    filtered_gene_list: list = None):
        """
        Training DeSide model

        :param training_set_file_path: the file path of training set, .h5ad file, log2cpm1p format, samples by genes
        :param hyper_params: pre-determined hyperparameters for DeSide model
        :param cell_types: specific a list of cell types instead of using all cell types in training set
        :param scaling_by_sample: whether to scale the expression values of each sample to [0, 1] by 'min_max'
        :param callback: whether to use callback function when training model
        :param n_epoch: the max number of epochs to train
        :param metrics: mse (regression model) / accuracy (classifier)
        :param n_patience: patience in early_stopping_callback
        :param remove_cancer_cell: remove cancer cell from y if True, using "1-others"
        :param fine_tune: fine tune pre-trained model
        :param scaling_by_constant: scaling GEP by dividing a constant in log space, default value is 20,
            to make sure all expression values are in [0, 1) if True
        :param one_minus_alpha: use 1 - alpha for all cell types if True
        :param verbose: whether to print progress during training, 0: silent, 1: progress bar, 2: one line per epoch
        :param pathway_mask: the mask of pathway genes, 1: pathway gene, 0: non-pathway gene, genes by pathways
        :param method_adding_pathway: the method to use pathway profiles, 'add_to_end' or 'convert'
        :param input_gene_list: the gene list used as input,
            if None: use all genes in training set;
            if "intersection_with_pathway_genes": use the intersection of genes in training set and genes in pathways;
            if "filtered_genes": use the genes in filtered_gene_list.
        :param filtered_gene_list: the list of genes used as input, if None, use all genes in training set
        """
        self.one_minus_alpha = one_minus_alpha
        if not os.path.exists(self.model_file_path):
            print_msg('Start to training model...', log_file_path=self.log_file_path)
            learning_rate = hyper_params['learning_rate']
            loss_function_alpha = hyper_params['loss_function_alpha']
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
            del x
            if len(training_set_file_path) >= 2:
                # if multiple training sets exist, re-normalise to TPM, in case some genes were removed during merging
                x_obj.to_tpm()
                x_obj.to_log2cpm1p()

            # get pathway profiles here
            if pathway_mask is not None:
                if input_gene_list == "intersection_with_pathway_genes":
                    gep_gene_list = [i for i in x_obj.exp.columns.to_list() if i in pathway_mask.index.to_list()]
                    # normalize to TPM when using intersection genes with pathways
                    x_obj.align_with_gene_list(gene_list=gep_gene_list, fill_not_exist=True)
                elif input_gene_list == 'filtered_genes' and filtered_gene_list is not None:
                    gep_gene_list = filtered_gene_list.copy()
                else:
                    gep_gene_list = x_obj.exp.columns.to_list()
                pathway_profile_gene_list = x_obj.exp.columns.to_list()
                if method_adding_pathway == 'add_to_end':
                    # _gene_list = x_obj.exp.columns.to_list()
                    pd.DataFrame(gep_gene_list).to_csv(self.gene_list_for_gep_file_path, sep="\t")
                pd.DataFrame(pathway_profile_gene_list).to_csv(self.gene_list_for_pathway_profile_file_path, sep="\t")

                x_obj = self._get_pathway_profiles(x_obj, pathway_mask,
                                                   method=method_adding_pathway, filtered_gene_list=gep_gene_list)

            if scaling_by_sample:
                x_obj.do_scaling()
            if scaling_by_constant:
                # file_obj = ReadExp(_x, exp_type='log_space')
                x_obj.do_scaling_by_constant()
            x = x_obj.get_exp()  # a dataframe, samples by genes

            self.gene_list = x.columns.to_list()  # a list of all gene names

            # filtering cell types in cell fraction file, for example removing the cell fraction of cancer cell
            if cell_types is None:
                self.cell_types = y.columns.to_list()  # a list
            else:
                cell_type_provided_not_in_y = [i for i in cell_types if i not in y.columns.to_list()]
                assert len(cell_type_provided_not_in_y) == 0, 'Provided cell types (' + \
                                                              ', '.join(cell_type_provided_not_in_y) + \
                                                              ') are not in the training set.'
                self.cell_types = cell_types
            # print('   Using cell types:', self.cell_types)
            if remove_cancer_cell:
                self.cell_types = [i for i in self.cell_types if i != 'Cancer Cells']
            y = y.loc[:, self.cell_types]  # a dataframe, samples by cell types

            # Save features and cell types
            pd.DataFrame(self.cell_types).to_csv(self.cell_type_file_path, sep="\t")
            pd.DataFrame(self.gene_list).to_csv(self.gene_list_file_path, sep="\t")

            print(f'   Use the following cell types: {self.cell_types} ' + 'during training.')
            print(f'   The shape of X is: {x.shape}, (n_sample, n_gene)')
            print(f'   The shape of y is: {y.shape}, (n_sample, n_cell_type)')
            if not fine_tune:
                n_pathway = pathway_mask.shape[1] if pathway_mask is not None else 0
                input_shape = len(self.gene_list)
                n_gene = len(self.gene_list) - n_pathway
                if hyper_params['pathway_network']:
                    input_shape = n_gene
                self._build_model(input_shape=input_shape, output_shape=len(self.cell_types),
                                  hyper_params=hyper_params, n_pathway=n_pathway)
            if self.model is None:
                raise FileNotFoundError('pre-trained model should be assigned to self.model')
            opt = keras.optimizers.Adam(learning_rate=learning_rate)

            # loss_function = functools.partial(loss_fn_mae_rmse, alpha=loss_function_alpha)
            # loss_function.get_config = lambda: {'alpha': loss_function_alpha}
            print('   The following loss function will be used:', 'mae +', loss_function_alpha, '* rmse')
            monitor_metrics = ['mae', keras.metrics.RootMeanSquaredError()]
            if metrics not in ['mae', 'rmse', 'mse']:
                monitor_metrics.append(metrics)
            self.model.compile(optimizer=opt, loss=loss_fn_mae_rmse, metrics=monitor_metrics)
            print(self.model.summary())

            # training model
            pathway_network = hyper_params['pathway_network']
            x = get_x_by_pathway_network(x, pathway_network=pathway_network, pathway_mask=pathway_mask)
            if callback:
                # Stop training when a monitored metric has stopped improving.
                # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
                early_stopping_callback = keras.callbacks.EarlyStopping(
                    # patience=10,
                    patience=n_patience,
                    monitor='val_loss',
                    mode='min',
                    restore_best_weights=True)
                history = self.model.fit(x, y.values, epochs=n_epoch,
                                         batch_size=batch_size, verbose=verbose, validation_split=0.2,
                                         callbacks=[early_stopping_callback])
                # history = self.model.fit(x.values, y.values, epochs=n_epoch,
                #                          batch_size=batch_size, verbose=2)
            else:
                history = self.model.fit(x, y.values, epochs=n_epoch,
                                         batch_size=batch_size, verbose=verbose, validation_split=0.2)

            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            hist.to_csv(os.path.join(self.model_dir, 'history_reg.csv'))
            plot_loss(hist, output_dir=self.model_dir, y_label='loss_function')

            self.model.save(self.model_file_path)
            key_params_file_path = os.path.join(self.model_dir, 'key_params.txt')
            print(f'   Key parameters during model training will be saved in {key_params_file_path}.')
            self.save_params(key_params_file_path)
            print_msg('Training done.', log_file_path=self.log_file_path)
        else:
            print(f'Previous model existed: {self.model_file_path}')

    @staticmethod
    def _get_pathway_profiles(x_obj, pathway_mask: pd.DataFrame, method='add_to_end', filtered_gene_list=None):
        """
        :param x_obj: input gene expression matrix, a class of ReadExp
        :param pathway_mask: pathway mask
        :param method: 'convert' or 'add_to_end', convert to pathway profiles or add to the end of x
        :param filtered_gene_list: if not None, genes will be filtered by this list and normalised to TPM
            after getting pathway profiles
        :return: pathway profiles, a class of ReadExp
        """
        if x_obj.file_type == 'log_space':
            x_obj.to_tpm()
        x = x_obj.get_exp()
        common_genes = list(set(x.columns) & set(pathway_mask.index))
        print('common genes between training set and pathway mask:', len(common_genes))
        genes_only_in_x = list(set(x.columns) - set(pathway_mask.index))
        # add genes only in x to pathway mask as all zeros
        if len(genes_only_in_x) > 0:
            print('genes only in training set:', len(genes_only_in_x))
            pathway_mask = pd.concat([pathway_mask,
                                      pd.DataFrame(np.zeros((len(genes_only_in_x), pathway_mask.shape[1])),
                                                   index=genes_only_in_x, columns=pathway_mask.columns)])
        pathway_mask = pathway_mask.loc[x.columns, :]  # genes by pathways
        if method == 'convert':
            x = x @ pathway_mask  # get pathway profiles by matrix multiplication
        elif method == 'add_to_end':
            x_pathway_profiles = x @ pathway_mask  # (m by n) x  (n by p) = m by p
            if filtered_gene_list is not None:  # filter genes and normalise to TPM after getting pathway profiles
                intersect_genes = list(set(x.columns) & set(filtered_gene_list))
                if len(intersect_genes) != len(x.columns) or len(intersect_genes) != len(filtered_gene_list):
                    x_obj.align_with_gene_list(gene_list=filtered_gene_list, fill_not_exist=True)
                    x = x_obj.get_exp()
            x = pd.concat([x, x_pathway_profiles], axis=1)  # combine x and pathway profiles by column, m x (n + p)
        # log2 transform
        x = np.log2(x + 1)
        print('x shape:', x.shape)
        x_obj = ReadExp(x, exp_type='log_space')
        return x_obj

    def get_x_before_predict(self, input_file, exp_type, transpose: bool = False, print_info: bool = True,
                             scaling_by_sample: bool = False, scaling_by_constant: bool = True,
                             pathway_mask: pd.DataFrame = None, method_adding_pathway: str = 'add_to_end'):
        """
        :param input_file: input file path
        :param exp_type: 'log_space' or 'raw_space'
        :param transpose: if True, transpose the input dataframe
        :param print_info: if True, print info
        :param scaling_by_sample: if True, scaling by sample
        :param scaling_by_constant: if True, scaling by constant
        :param pathway_mask: if not None, use pathway mask to get pathway profiles
        :param method_adding_pathway: 'add_to_end' or 'convert'
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

        if pathway_mask is not None:
            # get gene list for constructing pathway profiles
            gene_list_for_pathway_profile = self.get_gene_list_for_pathway_profile()
            gene_list_for_gep = None
            if method_adding_pathway == 'add_to_end':
                gene_list_for_gep = self.get_gene_list_for_gep()
            # check whether gene list for pathway profile is the same as the gene list in current input file
            intersection_genes = list(set(gene_list_for_pathway_profile) & set(read_df_obj.exp.columns.to_list()))
            if len(intersection_genes) != len(gene_list_for_pathway_profile) or \
                    len(intersection_genes) != len(read_df_obj.exp.columns.to_list()):
                read_df_obj.align_with_gene_list(gene_list=gene_list_for_pathway_profile, fill_not_exist=True)
            print(f'   {read_df_obj.exp.shape[1]} genes will be used to construct the pathway profiles.')
            read_df_obj = self._get_pathway_profiles(read_df_obj, pathway_mask, method=method_adding_pathway,
                                                     filtered_gene_list=gene_list_for_gep)

        # check gene list / pathway list
        pathway_list = True if pathway_mask is not None else False
        read_df_obj.align_with_gene_list(gene_list=self.gene_list, fill_not_exist=True, pathway_list=pathway_list)
        if pathway_mask is None:
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

    def predict(self, input_file, exp_type, output_file_path: str = None, transpose: bool = False,
                print_info: bool = True, add_cell_type: bool = False, scaling_by_constant=False,
                scaling_by_sample=True, one_minus_alpha: bool = False, pathway_mask: pd.DataFrame = None,
                method_adding_pathway: str = 'add_to_end', hyper_params: dict = None):
        """
        Predicting cell proportions using pre-trained model.

        :param input_file: the file path of input file (.csv / .h5ad / pd.Dataframe), samples by genes
            simulated (or TCGA) bulk expression profiles, log2(TPM + 1) or TPM
        :param output_file_path: the file path to save prediction result
        :param exp_type: log_space or TPM, log_space means log2(TPM + 1)
        :param transpose: transpose if exp_file formed as genes (index) by samples (columns)
        :param print_info: print information during prediction
        :param add_cell_type: only True when predicting cell types using classification model
        :param scaling_by_constant: scaling log2(TPM + 1) by dividing 20
        :param scaling_by_sample: scaling by sample, same as Scaden
        :param one_minus_alpha: use 1 - alpha for all cell types if True
        :param pathway_mask: if not None, use pathway mask to get pathway profiles
        :param method_adding_pathway: 'add_to_end' or 'convert'
        :param hyper_params: hyper parameters for DNN model
        """
        self.one_minus_alpha = one_minus_alpha
        if print_info:
            print('   Start to predict cell fractions by pre-trained model...')
        if self.cell_types is None:
            self.cell_types = self.get_cell_type()

        # load input data
        x = self.get_x_before_predict(input_file, exp_type, transpose=transpose, print_info=print_info,
                                      scaling_by_constant=scaling_by_constant, scaling_by_sample=scaling_by_sample,
                                      pathway_mask=pathway_mask, method_adding_pathway=method_adding_pathway)

        # load pre-trained model
        if self.model is None:
            try:
                self.model = keras.models.load_model(self.model_file_path)
            except ValueError:
                custom_objects = {'loss_fn_mae_rmse': loss_fn_mae_rmse}
                # with keras.saving.custom_object_scope(custom_objects):
                self.model = keras.models.load_model(self.model_file_path, custom_objects=custom_objects)
            finally:
                print(f'   Pre-trained model loaded from {self.model_file_path}.')
        pathway_network = hyper_params['pathway_network']
        x_index = x.index.copy()
        x = get_x_by_pathway_network(x, pathway_network=pathway_network, pathway_mask=pathway_mask)
        # predict using loaded model
        pred_result = self.model.predict(x)
        pred_df = pd.DataFrame(pred_result, index=x_index, columns=self.cell_types)
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
            pred_df['pred_cell_type'] = self._pred_cell_type_by_cell_frac(pred_cell_frac=pred_df)
        if print_info:
            print('   Model prediction done.')
        if output_file_path is not None:
            pred_df.to_csv(output_file_path, float_format='%.3f')
        else:
            return pred_df

    def get_model(self):
        """
        Load pre-trained model by `keras.models.load_model` if exists.

        :return: pre-trained model
        """
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
        """
        Get key parameters of the model.
        """
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

    def get_gene_list_for_gep(self) -> list:
        gene_list_for_gep = []
        if os.path.exists(self.gene_list_for_gep_file_path):
            gene_list_for_gep = list(pd.read_csv(self.gene_list_for_gep_file_path,
                                                 sep='\t', index_col=0)['0'])
        return gene_list_for_gep

    def get_gene_list_for_pathway_profile(self) -> list:
        gene_list_for_pathway_profile = []
        if os.path.exists(self.gene_list_for_pathway_profile_file_path):
            gene_list_for_pathway_profile = list(pd.read_csv(self.gene_list_for_pathway_profile_file_path,
                                                             sep='\t', index_col=0)['0'])
        return gene_list_for_pathway_profile

    def get_cell_type(self) -> list:
        if (self.cell_types is None) and os.path.exists(self.cell_type_file_path):
            self.cell_types = list(pd.read_csv(self.cell_type_file_path, sep='\t', index_col=0)['0'])
        return self.cell_types

    def save_params(self, output_file_path: str):
        key_params = self.get_parameters()
        with open(output_file_path, 'w') as f_handle:
            json.dump(key_params, fp=f_handle, indent=2)

    def _pred_cell_type_by_cell_frac(self, pred_cell_frac: pd.DataFrame) -> list:
        """
        convert predicted cell fractions to cell types
        """
        id2cell_type = {i: self.cell_types[i] for i in range(len(self.cell_types))}
        pred_id = pred_cell_frac.values.argmax(axis=1)
        return [id2cell_type[i] for i in pred_id]


def loss_fn_mae_rmse(y_true, y_pred, alpha=0.5):
    """
    Customized loss function for training the model. `alpha*MAE + (1-alpha)*RMSE`

    :param y_true: true cell fractions
    :param y_pred: predicted cell fractions
    :param alpha: weight of MAE
    """
    mae = keras.losses.MeanAbsoluteError()
    mse = keras.losses.MeanSquaredError()
    return alpha * mae(y_true, y_pred) + (1 - alpha) * tf.sqrt(mse(y_true, y_pred))
