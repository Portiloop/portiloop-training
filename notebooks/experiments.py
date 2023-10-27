from portiloopml.portiloop_python.ANN.portiloop_detector_training import train
from portiloopml.portiloop_python.ANN.models.test_models import PortiConvAtt
from portiloopml.portiloop_python.ANN.utils import LoggerWandb, get_configs, set_seeds
from portiloopml.portiloop_python.ANN.data.moda_data import generate_dataloader
from portiloopml.portiloop_python.ANN.data.mass_data import generate_entire_dataset_MASS, get_dataloaders_mass, get_dataloaders_sleep_stage, read_pretraining_dataset, read_sleep_staging_labels, read_spindle_trains_labels
from portiloopml.portiloop_python.ANN.models.test_models import PortiResNet
from torchsummary import summary
from portiloopml.portiloop_python.ANN.models.lstm import PortiloopNetwork

config = get_configs("Test", False, 0)
config['batch_size_validation'] = 64

print("Reading filters data...")
filt_data = read_pretraining_dataset(config['MASS_dir'])
print("Done...")
labels = read_spindle_trains_labels(config['old_dataset'])
ss_labels = read_sleep_staging_labels(config['path_dataset'])
subjects = list(filt_data.keys())
del filt_data

subsets = ['01', '02', '03', '05']
for subset in subsets:
    print(f"Generating subset {subset}...")
    subjects_ss = [subject for subject in subjects if subject[3:].startswith(subset)]
    generate_entire_dataset_MASS(subjects_ss, labels, ss_labels, subset)
exit()

print("Setting up config...")
experiment_name = 'testing_mass_training_again'
test_set = True
seed = 42
set_seeds(seed)
config_dict = get_configs(experiment_name, test_set, seed)
config_dict['hidden_size'] = 64
config_dict['nb_rnn_layers'] = 3
config_dict['after_rnn'] = 'hidden'
print("Done...")

print("Loading data...")
# config_dict['batch_size_validation'] = 4096
# config_dict['seq_len'] = 1
# config_dict['batch_size'] = 64
# config_dict['window_size'] = 250
# train_loader, val_loader, batch_size_validation, _, _, _ = generate_dataloader(
#     config_dict)
# config_dict['batch_size_validation'] = batch_size_validation
config_dict['batch_size_validation'] = 256

# train_loader, val_loader = get_dataloaders_mass(config_dict)

# print(next(iter(val_loader))[0].shape)
# train_loader, val_loader = get_dataloaders_sleep_stage(config_dict)

print("Done...")

print("Initializing model...")
model = PortiloopNetwork(config_dict)
device = config_dict["device_train"]
model.to(device)
summary(model, [(config_dict['seq_len'], 1, config_dict['window_size']),
        (config_dict['nb_rnn_layers'], config_dict['batch_size'],
         config_dict['hidden_size'])])

depth = 5
model = PortiResNet(
    depth, config_dict['hidden_size'], config_dict['nb_rnn_layers'])
summary(model)
model.to(config_dict["device_train"])
print("Done...")

print("Starting training...")
recurrent = True
save_model = True
unique_name = False
# config_dict['lr_adam'] = 0.0005

print("Starting logger...")
wandb_project = f"Portiloop"
wandb_group = 'new_training'
logger = LoggerWandb(experiment_name, config_dict,
                     wandb_project, group=wandb_group)
print("Done...")

# train(
#     train_loader,
#     val_loader,
#     model,
#     recurrent,
#     logger,
#     save_model,
#     unique_name,
#     config_dict
# )
print("Done...")
