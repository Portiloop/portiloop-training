import portiloop_detector_training as portiloop

exp_index = 0
experiment_name = "run_v2_BEST_3"

filename_dataset = "13042021_portiloop_dataset_250_standardized_envelope_pf_labeled.txt"

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

samp_test = portiloop.ValidationSampler(ds_test,
                                        nb_samples=int(len(ds_test) / max(seq_stride, portiloop.div_val_samp)),
                                        seq_stride=seq_stride)

test_loader = portiloop.DataLoader(ds_test,
                                   batch_size=1,
                                   sampler=samp_test,
                                   num_workers=0,
                                   pin_memory=True,
                                   shuffle=False)

criterion = portiloop.nn.MSELoss()

accuracy_test, loss_test, f1_test, precision_test, recall_test = portiloop.run_inference(
    test_loader, criterion, net, device_val, hidden_size, nb_rnn_layers)

print(f"accuracy_test = {accuracy_test}")
print(f"loss_test = {loss_test}")
print(f"f1_test = {f1_test}")
print(f"precision_test = {precision_test}")
print(f"recall_test = {recall_test}")
