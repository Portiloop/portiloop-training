import numpy as np

import portiloop_detector_training as portiloop

exp_index = 0
experiment_name = "run_v1_3080_20210413085714_0"

filename_dataset = "0908_portiloop_dataset_small_250_standardized_simulink_envelope_pf_labeled.txt"

config_dict = portiloop.get_config_dict(exp_index, experiment_name)

net = portiloop.PortiloopNetwork(config_dict)

checkpoint = portiloop.torch.load(portiloop.path_dataset / experiment_name)
net.load_state_dict(checkpoint['model_state_dict'])

fe = config_dict["fe"]

nb_epoch_max = config_dict["nb_epoch_max"]
nb_batch_per_epoch = config_dict["nb_batch_per_epoch"]
nb_epoch_early_stopping_stop = config_dict["nb_epoch_early_stopping_stop"]
early_stopping_smoothing_factor = config_dict["early_stopping_smoothing_factor"]
batch_size = config_dict["batch_size"]
seq_len = config_dict["seq_len"]
window_size_s = config_dict["window_size_s"]
seq_stride_s = config_dict["seq_stride_s"]
lr_adam = config_dict["lr_adam"]
hidden_size = config_dict["hidden_size"]
device_val = config_dict["device_val"]
device_train = config_dict["device_train"]
max_duration = config_dict["max_duration"]
nb_rnn_layers = config_dict["nb_rnn_layers"]
adam_w = config_dict["adam_w"]
window_size = int(window_size_s * fe)
seq_stride = int(seq_stride_s * fe)

ds_test = portiloop.SignalDataset(filename=filename_dataset,
                                  path=portiloop.path_dataset,
                                  window_size=window_size,
                                  fe=fe,
                                  seq_len=1,
                                  start_ratio=0,
                                  end_ratio=1)

criterion = portiloop.nn.MSELoss()

THRESHOLD = 0.4

state = []
h1 = portiloop.torch.zeros((nb_rnn_layers, 1, hidden_size), device=device_val)
h2 = portiloop.torch.zeros((nb_rnn_layers, 1, hidden_size), device=device_val)
last = 0

with portiloop.torch.no_grad():
    for idx in range(0, 250 * 60*45, seq_stride):
        batch_samples_input1, batch_samples_input2, batch_samples_input3, batch_labels = ds_test[idx]
        batch_samples_input1 = batch_samples_input1.to(device=device_val).float().view(1, 1, -1)
        batch_samples_input2 = batch_samples_input2.to(device=device_val).float().view(1, 1, -1)
        batch_samples_input3 = batch_samples_input3.to(device=device_val).float().view(1, 1, -1)
        batch_labels = batch_labels.to(device=device_val).float()
        output, h1, h2 = net(batch_samples_input1, batch_samples_input2, batch_samples_input3, h1, h2)
        output = output.view(-1)

        if output > THRESHOLD:
            if batch_labels > THRESHOLD:
                state.append(1)
            else:
                state.append(2)
        else:
            if batch_labels > THRESHOLD:
                state.append(3)
            else:
                state.append(4)

np.savetxt(portiloop.path_dataset / "labels_portiloop.txt", state)
#
# plt.cla()
#
# for i, st in enumerate(state):
#     color = 'w'
#     label = "Not evaluated"
#     idx = i * seq_stride
#     if st == 1:
#         color = 'g'
#         label = "True Positive"
#     elif st == 2:
#         color = 'r'
#         label = "False Positive"
#     elif st == 4:
#         color = 'b'
#         label = "True Negative"
#     elif st == 3:
#         color = 'k'
#         label = "False Negative"
#     plt.plot(np.arange(idx, idx + seq_stride) / 250, ds_test.full_signal[idx:idx + seq_stride].detach().numpy(), linewidth=1, color=color)
# plt.legend(loc='upper left')
# plt.xlabel("time (s)")
#
# plt.ylim([-20, 20])
# plt.xlim([20, 50])
# plt.show()
