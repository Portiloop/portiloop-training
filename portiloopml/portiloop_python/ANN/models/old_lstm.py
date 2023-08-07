# class ConvPoolModule(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channel,
#                  kernel_conv,
#                  stride_conv,
#                  conv_padding,
#                  dilation_conv,
#                  kernel_pool,
#                  stride_pool,
#                  pool_padding,
#                  dilation_pool,
#                  dropout_p):
#         super(ConvPoolModule, self).__init__()

#         self.conv = nn.Conv1d(in_channels=in_channels,
#                               out_channels=out_channel,
#                               kernel_size=kernel_conv,
#                               stride=stride_conv,
#                               padding=conv_padding,
#                               dilation=dilation_conv)
#         self.pool = nn.MaxPool1d(kernel_size=kernel_pool,
#                                  stride=stride_pool,
#                                  padding=pool_padding,
#                                  dilation=dilation_pool)
#         self.dropout = nn.Dropout(dropout_p)

#     def forward(self, input_f):
#         x, max_value = input_f
#         x = F.relu(self.conv(x))
#         x = self.pool(x)
#         max_temp = torch.max(abs(x))
#         if max_temp > max_value:
#             logging.debug(f"max_value = {max_temp}")
#             max_value = max_temp
#         return self.dropout(x), max_value


# class FcModule(nn.Module):
#     def __init__(self,
#                  in_features,
#                  out_features,
#                  dropout_p):
#         super(FcModule, self).__init__()

#         self.fc = nn.Linear(in_features=in_features, out_features=out_features)
#         self.dropout = nn.Dropout(dropout_p)

#     def forward(self, x):
#         x = F.relu(self.fc(x))
#         return self.dropout(x)


# class PortiloopNetwork(nn.Module):
#     def __init__(self, c_dict):
#         super(PortiloopNetwork, self).__init__()

#         RNN = c_dict["RNN"]
#         stride_pool = c_dict["stride_pool"]
#         stride_conv = c_dict["stride_conv"]
#         kernel_conv = c_dict["kernel_conv"]
#         kernel_pool = c_dict["kernel_pool"]
#         nb_channel = c_dict["nb_channel"]
#         hidden_size = c_dict["hidden_size"]
#         window_size_s = c_dict["window_size_s"]
#         dropout_p = c_dict["dropout"]
#         dilation_conv = c_dict["dilation_conv"]
#         dilation_pool = c_dict["dilation_pool"]
#         fe = c_dict["fe"]
#         nb_conv_layers = c_dict["nb_conv_layers"]
#         nb_rnn_layers = c_dict["nb_rnn_layers"]
#         first_layer_dropout = c_dict["first_layer_dropout"]
#         self.envelope_input = c_dict["envelope_input"]
#         self.power_features_input = c_dict["power_features_input"]
#         self.classification = c_dict["classification"]

#         conv_padding = 0  # int(kernel_conv // 2)
#         pool_padding = 0  # int(kernel_pool // 2)
#         window_size = int(window_size_s * fe)
#         nb_out = window_size

#         for _ in range(nb_conv_layers):
#             nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)
#             nb_out = out_dim(nb_out, pool_padding, dilation_pool, kernel_pool, stride_pool)

#         output_cnn_size = int(nb_channel * nb_out)

#         self.RNN = RNN
#         self.first_layer_input1 = ConvPoolModule(in_channels=1,
#                                                  out_channel=nb_channel,
#                                                  kernel_conv=kernel_conv,
#                                                  stride_conv=stride_conv,
#                                                  conv_padding=conv_padding,
#                                                  dilation_conv=dilation_conv,
#                                                  kernel_pool=kernel_pool,
#                                                  stride_pool=stride_pool,
#                                                  pool_padding=pool_padding,
#                                                  dilation_pool=dilation_pool,
#                                                  dropout_p=dropout_p if first_layer_dropout else 0)
#         self.seq_input1 = nn.Sequential(*(ConvPoolModule(in_channels=nb_channel,
#                                                          out_channel=nb_channel,
#                                                          kernel_conv=kernel_conv,
#                                                          stride_conv=stride_conv,
#                                                          conv_padding=conv_padding,
#                                                          dilation_conv=dilation_conv,
#                                                          kernel_pool=kernel_pool,
#                                                          stride_pool=stride_pool,
#                                                          pool_padding=pool_padding,
#                                                          dilation_pool=dilation_pool,
#                                                          dropout_p=dropout_p) for _ in range(nb_conv_layers - 1)))
#         if RNN:
#             self.gru_input1 = nn.GRU(input_size=output_cnn_size,
#                                      hidden_size=hidden_size,
#                                      num_layers=nb_rnn_layers,
#                                      dropout=0,
#                                      batch_first=True)
#         #       fc_size = hidden_size
#         else:
#             self.first_fc_input1 = FcModule(in_features=output_cnn_size, out_features=hidden_size, dropout_p=dropout_p)
#             self.seq_fc_input1 = nn.Sequential(
#                 *(FcModule(in_features=hidden_size, out_features=hidden_size, dropout_p=dropout_p) for _ in range(nb_rnn_layers - 1)))
#         if self.envelope_input:
#             self.first_layer_input2 = ConvPoolModule(in_channels=1,
#                                                      out_channel=nb_channel,
#                                                      kernel_conv=kernel_conv,
#                                                      stride_conv=stride_conv,
#                                                      conv_padding=conv_padding,
#                                                      dilation_conv=dilation_conv,
#                                                      kernel_pool=kernel_pool,
#                                                      stride_pool=stride_pool,
#                                                      pool_padding=pool_padding,
#                                                      dilation_pool=dilation_pool,
#                                                      dropout_p=dropout_p if first_layer_dropout else 0)
#             self.seq_input2 = nn.Sequential(*(ConvPoolModule(in_channels=nb_channel,
#                                                              out_channel=nb_channel,
#                                                              kernel_conv=kernel_conv,
#                                                              stride_conv=stride_conv,
#                                                              conv_padding=conv_padding,
#                                                              dilation_conv=dilation_conv,
#                                                              kernel_pool=kernel_pool,
#                                                              stride_pool=stride_pool,
#                                                              pool_padding=pool_padding,
#                                                              dilation_pool=dilation_pool,
#                                                              dropout_p=dropout_p) for _ in range(nb_conv_layers - 1)))

#             if RNN:
#                 self.gru_input2 = nn.GRU(input_size=output_cnn_size,
#                                          hidden_size=hidden_size,
#                                          num_layers=nb_rnn_layers,
#                                          dropout=0,
#                                          batch_first=True)
#             else:
#                 self.first_fc_input2 = FcModule(in_features=output_cnn_size, out_features=hidden_size, dropout_p=dropout_p)
#                 self.seq_fc_input2 = nn.Sequential(
#                     *(FcModule(in_features=hidden_size, out_features=hidden_size, dropout_p=dropout_p) for _ in range(nb_rnn_layers - 1)))
#         fc_features = 0
#         fc_features += hidden_size
#         if self.envelope_input:
#             fc_features += hidden_size
#         if self.power_features_input:
#             fc_features += 1
#         out_features = 1
#         self.fc = nn.Linear(in_features=fc_features,  # enveloppe and signal + power features ratio
#                             out_features=out_features)  # probability of being a spindle

#     def forward(self, x1, x2, x3, h1, h2, max_value=np.inf):
#         # x1 : input 1 : cleaned signal
#         # x2 : input 2 : envelope
#         # x3 : power features ratio
#         # h1 : gru 1 hidden size
#         # h2 : gru 2 hidden size
#         # max_value (optional) : print the maximal value reach during inference (used to verify if the FPGA implementation precision is enough)
#         (batch_size, sequence_len, features) = x1.shape

#         if ABLATION == 1:
#             x1 = copy.deepcopy(x2)
#         elif ABLATION == 2:
#             x2 = copy.deepcopy(x1)

#         x1 = x1.view(-1, 1, features)
#         x1, max_value = self.first_layer_input1((x1, max_value))
#         x1, max_value = self.seq_input1((x1, max_value))

#         x1 = torch.flatten(x1, start_dim=1, end_dim=-1)
#         hn1 = None
#         if self.RNN:
#             x1 = x1.view(batch_size, sequence_len, -1)
#             x1, hn1 = self.gru_input1(x1, h1)
#             max_temp = torch.max(abs(x1))
#             if max_temp > max_value:
#                 logging.debug(f"max_value = {max_temp}")
#                 max_value = max_temp
#             x1 = x1[:, -1, :]
#         else:
#             x1 = self.first_fc_input1(x1)
#             x1 = self.seq_fc_input1(x1)
#         x = x1
#         hn2 = None
#         if self.envelope_input:
#             x2 = x2.view(-1, 1, features)
#             x2, max_value = self.first_layer_input2((x2, max_value))
#             x2, max_value = self.seq_input2((x2, max_value))

#             x2 = torch.flatten(x2, start_dim=1, end_dim=-1)
#             if self.RNN:
#                 x2 = x2.view(batch_size, sequence_len, -1)
#                 x2, hn2 = self.gru_input2(x2, h2)
#                 max_temp = torch.max(abs(x2))
#                 if max_temp > max_value:
#                     logging.debug(f"max_value = {max_temp}")
#                     max_value = max_temp
#                 x2 = x2[:, -1, :]
#             else:
#                 x2 = self.first_fc_input2(x2)
#                 x2 = self.seq_fc_input2(x2)
#             x = torch.cat((x, x2), -1)

#         if self.power_features_input:
#             x3 = x3.view(-1, 1)
#             x = torch.cat((x, x3), -1)

#         x = self.fc(x)  # output size: 1
#         max_temp = torch.max(abs(x))
#         if max_temp > max_value:
#             logging.debug(f"max_value = {max_temp}")
#             max_value = max_temp
#         x = torch.sigmoid(x)

#         return x, hn1, hn2, max_value

# def get_config_dict(index, split_i):
#     # config_dict = {'experiment_name': f'pareto_search_10_619_{index}', 'device_train': 'cuda:0', 'device_val': 'cuda:0', 'nb_epoch_max': 1000,
#     # 'max_duration': 257400, 'nb_epoch_early_stopping_stop': 20, 'early_stopping_smoothing_factor': 0.1, 'fe': 250, 'nb_batch_per_epoch': 5000,
#     # 'first_layer_dropout': False, 'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
#     # 'nb_conv_layers': 3, 'seq_len': 50, 'nb_channel': 16, 'hidden_size': 32, 'seq_stride_s': 0.08600000000000001, 'nb_rnn_layers': 1,
#     # 'RNN': True, 'envelope_input': True, 'window_size_s': 0.266, 'stride_pool': 1, 'stride_conv': 1, 'kernel_conv': 9, 'kernel_pool': 7,
#     # 'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 24, 'time_in_past': 4.300000000000001, 'estimator_size_memory': 1628774400, "batch_size":
#     # batch_size_list[index % len(batch_size_list)], "lr_adam": lr_adam_list[index % len(lr_adam_list)]}
#     c_dict = {'experiment_name': f'spindleNet_{index}', 'device_train': 'cuda:0', 'device_val':
#         'cuda:0', 'nb_epoch_max': 500,
#               'max_duration': 257400, 'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250,
#               'nb_batch_per_epoch': 1000,
#               'first_layer_dropout': False,
#               'power_features_input': True, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
#               'reg_balancing': 'none',
#               'nb_conv_layers': 5,
#               'seq_len': 50, 'nb_channel': 40, 'hidden_size': 100, 'seq_stride_s': 0.004, 'nb_rnn_layers': 1, 'RNN': True,
#               'envelope_input': True,
#               "batch_size": 20, "lr_adam": 0.0009,
#               'window_size_s': 0.250, 'stride_pool': 1, 'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 5,
#               'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 2, 'time_in_past': 1.55, 'estimator_size_memory': 139942400}
#     # put LSTM and Softmax for the occasion and add padding, not exactly the same frequency (spindleNet = 200 Hz)

#     c_dict = {'experiment_name': f'ABLATION_{ABLATION}_test_v11_implemented_on_portiloop_{index}', 'device_train': 'cuda:0', 'device_val':
#         'cuda:0', 'nb_epoch_max': 500,
#               'max_duration': 257400, 'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250,
#               'nb_batch_per_epoch': 1000,
#               'first_layer_dropout': False,
#               'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
#               'reg_balancing': 'none',
#               'nb_conv_layers': 4,
#               'seq_len': 50, 'nb_channel': 26, 'hidden_size': 7, 'seq_stride_s': 0.044, 'nb_rnn_layers': 2, 'RNN': True,
#               'envelope_input': True,
#               "batch_size": 256, "lr_adam": 0.0009,
#               'window_size_s': 0.234, 'stride_pool': 1, 'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 9,
#               'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 2, 'time_in_past': 1.55, 'estimator_size_memory': 139942400,
#               'split_idx': split_i, 'validation_network_stride': 1}
#     c_dict = {'experiment_name': f'pareto_search_15_35_v5_small_seq_{index}', 'device_train': 'cuda:0', 'device_val': 'cuda:0', 'nb_epoch_max': 150,
#               'max_duration':
#                   257400,
#               'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250, 'nb_batch_per_epoch': 1000,
#               'first_layer_dropout': False,
#               'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
#               'reg_balancing': 'none',
#               'split_idx': split_i, 'validation_network_stride': 1, 'nb_conv_layers': 3, 'seq_len': 50, 'nb_channel': 31, 'hidden_size': 7,
#               'seq_stride_s': 0.02,
#               'nb_rnn_layers': 1, 'RNN': True, 'envelope_input': False, 'lr_adam': 0.0005, 'batch_size': 256, 'window_size_s': 0.218,
#               'stride_pool': 1,
#               'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 7, 'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 18, 'time_in_past': 8.5,
#               'estimator_size_memory': 188006400}
#     c_dict = {'experiment_name': f'ABLATION_{ABLATION}_2inputs_network_{index}', 'device_train': 'cuda:0', 'device_val':
#         'cuda:0', 'nb_epoch_max': 500,
#               'max_duration': 257400, 'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250,
#               'nb_batch_per_epoch': 1000,
#               'first_layer_dropout': False,
#               'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
#               'reg_balancing': 'none',
#               'nb_conv_layers': 4,
#               'seq_len': 50, 'nb_channel': 26, 'hidden_size': 7, 'seq_stride_s': 0.044, 'nb_rnn_layers': 2, 'RNN': True,
#               'envelope_input': True,
#               "batch_size": 256, "lr_adam": 0.0009,
#               'window_size_s': 0.234, 'stride_pool': 1, 'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 9,
#               'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 2, 'time_in_past': 1.55, 'estimator_size_memory': 139942400,
#               'split_idx': split_i, 'validation_network_stride': 1}
#     c_dict = {'experiment_name': f'pareto_search_15_35_v6_{index}', 'device_train': 'cuda:0', 'device_val': 'cuda:0', 'nb_epoch_max': 500,
#               'max_duration':
#                   257400,
#               'nb_epoch_early_stopping_stop': 100, 'early_stopping_smoothing_factor': 0.1, 'fe': 250, 'nb_batch_per_epoch': 1000,
#               'first_layer_dropout': False,
#               'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True,
#               'reg_balancing': 'none',
#               'split_idx': split_i, 'validation_network_stride': 1, 'nb_conv_layers': 3, 'seq_len': 50, 'nb_channel': 31, 'hidden_size': 7,
#               'seq_stride_s': 0.17,
#               'nb_rnn_layers': 1, 'RNN': True, 'envelope_input': False, 'lr_adam': 0.0005, 'batch_size': 256, 'window_size_s': 0.218,
#               'stride_pool': 1,
#               'stride_conv': 1, 'kernel_conv': 7, 'kernel_pool': 7, 'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 18, 'time_in_past': 8.5,
#               'estimator_size_memory': 188006400}
#     return c_dict
