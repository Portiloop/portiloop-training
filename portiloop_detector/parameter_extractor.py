import portiloop_detector_training as portiloop
import numpy as np

exp_index = 0
experiment_name = "run_v2_BEST_6"

config_dict = portiloop.get_config_dict(exp_index, experiment_name)

net = portiloop.PortiloopNetwork(config_dict)

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
    print(param.size())
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
    print(param.size())
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
file.close()
