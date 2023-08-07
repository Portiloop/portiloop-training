"""
This script extracts the weights of a trained ANN in pytorch so as to implement the ANN in FPGA.
"""

import portiloop_detector_training as portiloop


def float_to_string(f):
    return format(f, '.60g')


config_dict = portiloop.get_config_dict(29, 0)
net = portiloop.PortiloopNetwork(config_dict)
experiment_name = "pareto_search_15_35_v4_0"  # config_dict["experiment_name"]
checkpoint = portiloop.torch.load(portiloop.path_dataset / experiment_name)
net.load_state_dict(checkpoint['model_state_dict'])

file = open(portiloop.path_dataset / "weight.txt", 'w')
conv1_kernel_input1 = net.first_layer_input1.conv.weight.detach().numpy()
n, m, w = conv1_kernel_input1.shape

conv1_kernel_input1_str = "const parameter_fixed_type conv1_input1_kernel[CHANNEL_SIZE][1][CONV_K_SIZE] = {"
for i in range(n):
    conv1_kernel_input1_str += "{"
    for j in range(m):
        conv1_kernel_input1_str += "{"
        for k in range(w):
            conv1_kernel_input1_str += float_to_string(conv1_kernel_input1[i, j, k])
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

conv1_bias_input1_str = "const parameter_fixed_type conv1_input1_bias[CHANNEL_SIZE] = {"
for i in range(n):
    conv1_bias_input1_str += float_to_string(conv1_bias_input1[i])
    conv1_bias_input1_str += ", "
conv1_bias_input1_str = conv1_bias_input1_str[:-2]
conv1_bias_input1_str += "};\n"
file.write(conv1_bias_input1_str)

param_list = ["w_ih", "w_hh", "b_ih", "b_hh"]
param_name_list = ["WI", "WH", "B", "B"]
counter = 0
for param in net.gru_input1.parameters():
    param = param.view(-1).detach().numpy()
    n = len(param)

    input1_str = "const parameter_fixed_type gru" + str(counter // 4 + 1) + "_input1_" + param_list[counter % 4] + "[GRU_" + param_name_list[counter % 4] + "_SIZE] = {"
    for i in range(n):
        input1_str += float_to_string(param[i])
        input1_str += ", "
    input1_str = input1_str[:-2]
    input1_str += "};\n"
    file.write(input1_str)
    counter += 1

for layer in range(len(net.seq_input1)):
    conv2_kernel_input1 = net.seq_input1[layer].conv.weight.detach().numpy()
    n, m, w = conv2_kernel_input1.shape

    conv2_kernel_input1_str = "const parameter_fixed_type conv" + str(2 + layer) + "_input1_kernel[CHANNEL_SIZE][CHANNEL_SIZE][CONV_K_SIZE] = {"
    for i in range(n):
        conv2_kernel_input1_str += "{"
        for j in range(m):
            conv2_kernel_input1_str += "{"
            for k in range(w):
                conv2_kernel_input1_str += format(conv2_kernel_input1[i, j, k], '.60g')
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

    conv2_bias_input1_str = "const parameter_fixed_type conv" + str(2 + layer) + "_input1_bias[CHANNEL_SIZE] = {"
    for i in range(n):
        conv2_bias_input1_str += float_to_string(conv2_bias_input1[i])
        conv2_bias_input1_str += ", "
    conv2_bias_input1_str = conv2_bias_input1_str[:-2]
    conv2_bias_input1_str += "};\n"
    file.write(conv2_bias_input1_str)
if config_dict["envelope_input"]:

    conv1_kernel_input2 = net.first_layer_input2.conv.weight.detach().numpy()
    n, m, w = conv1_kernel_input2.shape

    conv1_kernel_input2_str = "const parameter_fixed_type conv1_input2_kernel[CHANNEL_SIZE][1][CONV_K_SIZE] = {"
    for i in range(n):
        conv1_kernel_input2_str += "{"
        for j in range(m):
            conv1_kernel_input2_str += "{"
            for k in range(w):
                conv1_kernel_input2_str += float_to_string(conv1_kernel_input2[i, j, k])
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

    conv1_bias_input2_str = "const parameter_fixed_type conv1_input2_bias[CHANNEL_SIZE] = {"
    for i in range(n):
        conv1_bias_input2_str += float_to_string(conv1_bias_input2[i])
        conv1_bias_input2_str += ", "
    conv1_bias_input2_str = conv1_bias_input2_str[:-2]
    conv1_bias_input2_str += "};\n"
    file.write(conv1_bias_input2_str)

    for layer in range(len(net.seq_input2)):
        conv2_kernel_input2 = net.seq_input2[layer].conv.weight.detach().numpy()
        n, m, w = conv2_kernel_input2.shape

        conv2_kernel_input2_str = "const parameter_fixed_type conv" + str(2 + layer) + "_input2_kernel[CHANNEL_SIZE][CHANNEL_SIZE][CONV_K_SIZE] = {"
        for i in range(n):
            conv2_kernel_input2_str += "{"
            for j in range(m):
                conv2_kernel_input2_str += "{"
                for k in range(w):
                    conv2_kernel_input2_str += float_to_string(conv2_kernel_input2[i, j, k])
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

        conv2_bias_input2_str = "const parameter_fixed_type conv" + str(2 + layer) + "_input2_bias[CHANNEL_SIZE] = {"
        for i in range(n):
            conv2_bias_input2_str += float_to_string(conv2_bias_input2[i])
            conv2_bias_input2_str += ", "
        conv2_bias_input2_str = conv2_bias_input2_str[:-2]
        conv2_bias_input2_str += "};\n"
        file.write(conv2_bias_input2_str)
    counter = 0
    for param in net.gru_input2.parameters():
        param = param.view(-1).detach().numpy()
        n = len(param)

        input2_str = "const parameter_fixed_type gru" + str(counter // 4 + 1) + "_input2_" + param_list[counter % 4] + "[GRU_" + param_name_list[counter % 4] + "_SIZE] = {"
        for i in range(n):
            input2_str += float_to_string(param[i])
            input2_str += ", "
        input2_str = input2_str[:-2]
        input2_str += "};\n"
        file.write(input2_str)
        counter += 1

param = net.fc.weight.view(-1).detach().numpy()
n = len(param)

fc_str = "const parameter_fixed_type fc_weight[FC_WEIGHT_SIZE] = {"
for i in range(n):
    fc_str += float_to_string(param[i])
    fc_str += ", "
fc_str = fc_str[:-2]
fc_str += "};\n"
file.write(fc_str)

param = net.fc.bias.detach().numpy()
n = len(param)

fc_str = "const parameter_fixed_type fc_bias[FC_OUTPUT_SIZE] = {"
for i in range(n):
    fc_str += float_to_string(param[i])
    fc_str += ", "
fc_str = fc_str[:-2]
fc_str += "};\n"
file.write(fc_str)

file.close()

for i in net.parameters():
    for j in i.view(-1):
        if abs(j) > 3:
            print(j)
