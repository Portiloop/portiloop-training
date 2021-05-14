import portiloop_detector_training as portiloop

config_dict = {'experiment_name': 'pareto_search_10_619', 'device_train': 'cuda:0', 'device_val': 'cuda:0', 'nb_epoch_max': 11, 'max_duration': 257400, 'nb_epoch_early_stopping_stop': 10, 'early_stopping_smoothing_factor': 0.1, 'fe': 250, 'nb_batch_per_epoch': 5000, 'batch_size': 256,
               'first_layer_dropout': False, 'power_features_input': False, 'dropout': 0.5, 'adam_w': 0.01, 'distribution_mode': 0, 'classification': True, 'nb_conv_layers': 3, 'seq_len': 50, 'nb_channel': 16, 'hidden_size': 32, 'seq_stride_s': 0.08600000000000001, 'nb_rnn_layers': 1,
               'RNN': True,
               'envelope_input': True, 'lr_adam': 0.0007, 'window_size_s': 0.266, 'stride_pool': 1, 'stride_conv': 1, 'kernel_conv': 9, 'kernel_pool': 7, 'dilation_conv': 1, 'dilation_pool': 1, 'nb_out': 24, 'time_in_past': 4.300000000000001, 'estimator_size_memory': 1628774400}
net = portiloop.PortiloopNetwork(config_dict)
experiment_name = "pareto_search_10_619"
checkpoint = portiloop.torch.load(portiloop.path_dataset / experiment_name)
net.load_state_dict(checkpoint['model_state_dict'])

file = open(portiloop.path_dataset / "weight.txt", 'w')
conv1_kernel_input1 = net.first_layer_input1.conv.weight.detach().numpy()
n, m, w = conv1_kernel_input1.shape

conv1_kernel_input1_str = "static const parameter_fixed_type conv1_input1_kernel[CONV1_C_OUT_SIZE][CONV1_C_IN_SIZE][CONV1_K_SIZE] = {"
for i in range(n):
    conv1_kernel_input1_str += "{"
    for j in range(m):
        conv1_kernel_input1_str += "{"
        for k in range(w):
            conv1_kernel_input1_str += str(conv1_kernel_input1[i, j, k])
            conv1_kernel_input1_str += ", "
        conv1_kernel_input1_str = conv1_kernel_input1_str[:-2]
        conv1_kernel_input1_str += "}, "
    conv1_kernel_input1_str = conv1_kernel_input1_str[:-2]
    conv1_kernel_input1_str += "}, "
conv1_kernel_input1_str = conv1_kernel_input1_str[:-2]
conv1_kernel_input1_str += "};\n"
file.write(conv1_kernel_input1_str)

conv1_bias_input1 = net.first_layer_input1.conv.bias.detach().numpy()
n = len(conv1_bias_input1)

conv1_bias_input1_str = "static const parameter_fixed_type conv1_input1_bias[CONV1_C_OUT_SIZE] = {"
for i in range(n):
    conv1_bias_input1_str += str(conv1_bias_input1[i])
    conv1_bias_input1_str += ", "
conv1_bias_input1_str = conv1_bias_input1_str[:-2]
conv1_bias_input1_str += "};\n"
file.write(conv1_bias_input1_str)

for layer in range(len(net.seq_input2)):
    conv2_kernel_input1 = net.seq_input1[layer].conv.weight.detach().numpy()
    n, m, w = conv2_kernel_input1.shape

    conv2_kernel_input1_str = "static const parameter_fixed_type conv" + str(2 + layer) + "_input1_kernel[CONV" + str(2 + layer) + "_C_OUT_SIZE][CONV" + str(2 + layer) + "_C_IN_SIZE][CONV" + str(2 + layer) + "_K_SIZE] = {"
    for i in range(n):
        conv2_kernel_input1_str += "{"
        for j in range(m):
            conv2_kernel_input1_str += "{"
            for k in range(w):
                conv2_kernel_input1_str += str(conv2_kernel_input1[i, j, k])
                conv2_kernel_input1_str += ", "
            conv2_kernel_input1_str = conv2_kernel_input1_str[:-2]
            conv2_kernel_input1_str += "}, "
        conv2_kernel_input1_str = conv2_kernel_input1_str[:-2]
        conv2_kernel_input1_str += "}, "
    conv2_kernel_input1_str = conv2_kernel_input1_str[:-2]
    conv2_kernel_input1_str += "};\n"
    file.write(conv2_kernel_input1_str)

    conv2_bias_input1 = net.seq_input1[layer].conv.bias.detach().numpy()
    n = len(conv2_bias_input1)

    conv2_bias_input1_str = "static const parameter_fixed_type conv" + str(2 + layer) + "_input1_bias[CONV" + str(2 + layer) + "_C_OUT_SIZE] = {"
    for i in range(n):
        conv2_bias_input1_str += str(conv2_bias_input1[i])
        conv2_bias_input1_str += ", "
    conv2_bias_input1_str = conv2_bias_input1_str[:-2]
    conv2_bias_input1_str += "};\n"
    file.write(conv2_bias_input1_str)

conv1_kernel_input2 = net.first_layer_input2.conv.weight.detach().numpy()
n, m, w = conv1_kernel_input2.shape

conv1_kernel_input2_str = "static const parameter_fixed_type conv1_input2_kernel[CONV1_C_OUT_SIZE][CONV1_C_IN_SIZE][CONV1_K_SIZE] = {"
for i in range(n):
    conv1_kernel_input2_str += "{"
    for j in range(m):
        conv1_kernel_input2_str += "{"
        for k in range(w):
            conv1_kernel_input2_str += str(conv1_kernel_input2[i, j, k])
            conv1_kernel_input2_str += ", "
        conv1_kernel_input2_str = conv1_kernel_input2_str[:-2]
        conv1_kernel_input2_str += "}, "
    conv1_kernel_input2_str = conv1_kernel_input2_str[:-2]
    conv1_kernel_input2_str += "}, "
conv1_kernel_input2_str = conv1_kernel_input2_str[:-2]
conv1_kernel_input2_str += "};\n"
file.write(conv1_kernel_input2_str)

conv1_bias_input2 = net.first_layer_input2.conv.bias.detach().numpy()
n = len(conv1_bias_input2)

conv1_bias_input2_str = "static const parameter_fixed_type conv1_input2_bias[CONV1_C_OUT_SIZE] = {"
for i in range(n):
    conv1_bias_input2_str += str(conv1_bias_input2[i])
    conv1_bias_input2_str += ", "
conv1_bias_input2_str = conv1_bias_input2_str[:-2]
conv1_bias_input2_str += "};\n"
file.write(conv1_bias_input2_str)

for layer in range(len(net.seq_input2)):
    conv2_kernel_input2 = net.seq_input2[layer].conv.weight.detach().numpy()
    n, m, w = conv2_kernel_input2.shape

    conv2_kernel_input2_str = "static const parameter_fixed_type conv" + str(2 + layer) + "_input2_kernel[CONV" + str(2 + layer) + "_C_OUT_SIZE][CONV" + str(2 + layer) + "_C_IN_SIZE][CONV" + str(2 + layer) + "_K_SIZE] = {"
    for i in range(n):
        conv2_kernel_input2_str += "{"
        for j in range(m):
            conv2_kernel_input2_str += "{"
            for k in range(w):
                conv2_kernel_input2_str += str(conv2_kernel_input2[i, j, k])
                conv2_kernel_input2_str += ", "
            conv2_kernel_input2_str = conv2_kernel_input2_str[:-2]
            conv2_kernel_input2_str += "}, "
        conv2_kernel_input2_str = conv2_kernel_input2_str[:-2]
        conv2_kernel_input2_str += "}, "
    conv2_kernel_input2_str = conv2_kernel_input2_str[:-2]
    conv2_kernel_input2_str += "};\n"
    file.write(conv2_kernel_input2_str)

    conv2_bias_input2 = net.seq_input2[layer].conv.bias.detach().numpy()
    n = len(conv2_bias_input2)

    conv2_bias_input2_str = "static const parameter_fixed_type conv" + str(2 + layer) + "_input2_bias[CONV" + str(2 + layer) + "_C_OUT_SIZE] = {"
    for i in range(n):
        conv2_bias_input2_str += str(conv2_bias_input2[i])
        conv2_bias_input2_str += ", "
    conv2_bias_input2_str = conv2_bias_input2_str[:-2]
    conv2_bias_input2_str += "};\n"
    file.write(conv2_bias_input2_str)

param_list = ["w_ih", "w_hh", "b_ih", "b_hh"]
param_name_list = ["WI", "WH", "B", "B"]
counter = 0
for param in net.gru_input1.parameters():
    param = param.view(-1).detach().numpy()
    n = len(param)

    input1_str = "static const parameter_fixed_type gru" + str(counter // 4 + 1) + "_input1_" + param_list[counter % 4] + "[GRU" + str(counter // 4 + 1) + "_" + param_name_list[counter % 4] + "_SIZE] = {"
    for i in range(n):
        input1_str += str(param[i])
        input1_str += ", "
    input1_str = input1_str[:-2]
    input1_str += "};\n"
    file.write(input1_str)
    counter += 1

counter = 0
for param in net.gru_input2.parameters():
    param = param.view(-1).detach().numpy()
    n = len(param)

    input2_str = "static const parameter_fixed_type gru" + str(counter // 4 + 1) + "_input2_" + param_list[counter % 4] + "[GRU" + str(counter // 4 + 1) + "_" + param_name_list[counter % 4] + "_SIZE] = {"
    for i in range(n):
        input2_str += str(param[i])
        input2_str += ", "
    input2_str = input2_str[:-2]
    input2_str += "};\n"
    file.write(input2_str)
    counter += 1

param = net.fc.weight.view(-1).detach().numpy()
n = len(param)

fc_str = "static const parameter_fixed_type fc_weight[FC_WEIGHT_SIZE] = {"
for i in range(n):
    fc_str += str(param[i])
    fc_str += ", "
fc_str = fc_str[:-2]
fc_str += "};\n"
file.write(fc_str)

param = net.fc.bias.detach().numpy()
n = len(param)

fc_str = "static const parameter_fixed_type fc_bias[FC_OUTPUT_SIZE] = {"
for i in range(n):
    fc_str += str(param[i])
    fc_str += ", "
fc_str = fc_str[:-2]
fc_str += "};\n"
file.write(fc_str)

file.close()

for i in net.parameters():
    for j in i.view(-1):
        if (abs(j) > 1):
            print(j)