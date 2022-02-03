
def get_nn_dict(config_name, ablation, index, split_i):

    if config_name == 'spindle_net':
        return {'experiment_name': f'spindleNet_{index}', 'device_train': 'cuda:0', 'device_val':
                'cuda:0', 'nb_epoch_max': 500,
                'max_duration': 257400, 'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250,
                'nb_batch_per_epoch': 1000,
                'first_layer_dropout': False,
                'power_features_input': True, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
                'reg_balancing': 'none',
                'nb_conv_layers': 5,
                'seq_len': 50, 'nb_channel': 40, 'hidden_size': 100, 'seq_stride_s': 0.004, 'nb_rnn_layers': 1, 'RNN': True,
                'envelope_input': True,
                "batch_size": 20, "lr_adam": 0.0009,
                'window_size_s': 0.250, 'stride_pool': 1, 'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 5,
                'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 2, 'time_in_past': 1.55, 'estimator_size_memory': 139942400}
        # put LSTM and Softmax for the occasion and add padding, not exactly the same frequency (spindleNet = 200 Hz)

    elif config_name == 'ablation_portiloop':
        return {'experiment_name': f'ABLATION_{ablation}_test_v11_implemented_on_portiloop_{index}', 'device_train': 'cuda:0', 'device_val':
                'cuda:0', 'nb_epoch_max': 500,
                'max_duration': 257400, 'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250,
                'nb_batch_per_epoch': 1000,
                'first_layer_dropout': False,
                'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
                'reg_balancing': 'none',
                'nb_conv_layers': 4,
                'seq_len': 50, 'nb_channel': 26, 'hidden_size': 7, 'seq_stride_s': 0.044, 'nb_rnn_layers': 2, 'RNN': True,
                'envelope_input': True,
                "batch_size": 256, "lr_adam": 0.0009,
                'window_size_s': 0.234, 'stride_pool': 1, 'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 9,
                'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 2, 'time_in_past': 1.55, 'estimator_size_memory': 139942400,
                'split_idx': split_i, 'validation_network_stride': 1}
    elif config_name == 'pareto_search_small_seq':
        return {'experiment_name': f'pareto_search_15_35_v5_small_seq_{index}', 'device_train': 'cuda:0', 'device_val': 'cuda:0', 'nb_epoch_max': 150,
                'max_duration':
                257400,
                'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250, 'nb_batch_per_epoch': 1000,
                'first_layer_dropout': False,
                'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
                'reg_balancing': 'none',
                'split_idx': split_i, 'validation_network_stride': 1, 'nb_conv_layers': 3, 'seq_len': 50, 'nb_channel': 31, 'hidden_size': 7,
                'seq_stride_s': 0.02,
                'nb_rnn_layers': 1, 'RNN': True, 'envelope_input': False, 'lr_adam': 0.0005, 'batch_size': 256, 'window_size_s': 0.218,
                'stride_pool': 1,
                'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 7, 'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 18, 'time_in_past': 8.5,
                'estimator_size_memory': 188006400}
    elif config_name == '2_inputs_network':
        return {'experiment_name': f'ABLATION_{ablation}_2inputs_network_{index}', 'device_train': 'cuda:0', 'device_val':
                'cuda:0', 'nb_epoch_max': 500,
                'max_duration': 257400, 'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250,
                'nb_batch_per_epoch': 1000,
                'first_layer_dropout': False,
                'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
                'reg_balancing': 'none',
                'nb_conv_layers': 4,
                'seq_len': 50, 'nb_channel': 26, 'hidden_size': 7, 'seq_stride_s': 0.044, 'nb_rnn_layers': 2, 'RNN': True,
                'envelope_input': True,
                "batch_size": 256, "lr_adam": 0.0009,
                'window_size_s': 0.234, 'stride_pool': 1, 'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 9,
                'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 2, 'time_in_past': 1.55, 'estimator_size_memory': 139942400,
                'split_idx': split_i, 'validation_network_stride': 1}
    elif config_name == 'pareto_search_v6':
        return {'experiment_name': f'pareto_search_15_35_v6_{index}', 'device_train': 'cpu', 'device_val': 'cpu', 'nb_epoch_max': 500,
                'max_duration':
                257400,
                'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250, 'nb_batch_per_epoch': 1000,
                'first_layer_dropout': False,
                'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
                'reg_balancing': 'none',
                'split_idx': split_i, 'validation_network_stride': 1, 'nb_conv_layers': 3, 'seq_len': 50, 'nb_channel': 31, 'hidden_size': 7,
                'seq_stride_s': 0.17,
                'nb_rnn_layers': 1, 'RNN': True, 'envelope_input': False, 'lr_adam': 0.0005, 'batch_size': 256, 'window_size_s': 0.218,
                'stride_pool': 1,
                'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 7, 'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 18, 'time_in_past': 8.5,
                'estimator_size_memory': 188006400}
    elif config_name == 'pareto_search_v4':
        return {'experiment_name': f'pareto_search_15_35_v4_{index}', 'device_train': 'cpu', 'device_val': 'cpu',
                'device_inference': 'cpu', 'nb_epoch_max': 150, 'max_duration': 257400,
                'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250,
                'nb_batch_per_epoch': 1000,
                'first_layer_dropout': False,
                'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0,
                'classification': True,
                'reg_balancing': 'none',
                'split_idx': split_i, 'validation_network_stride': 1, 'nb_conv_layers': 3, 'seq_len': 50,
                'nb_channel': 31, 'hidden_size': 7,
                'seq_stride_s': 0.170,
                'nb_rnn_layers': 1, 'RNN': True, 'envelope_input': False, 'lr_adam': 0.0005, 'batch_size': 256,
                'window_size_s': 0.218,
                'stride_pool': 1,
                'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 7, 'dilation_conv': 1, 'dilation_pool': 1,
                'nb_out': 18, 'time_in_past': 8.5,
                'estimator_size_memory': 188006400}
    else:
        raise ValueError(
            'Given config dict must either be a json file with .json extension or one of the default names of configuration.')
