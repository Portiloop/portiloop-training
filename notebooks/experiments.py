from portiloop_software.portiloop_python.ANN.portiloop_detector_training import train
from portiloop_software.portiloop_python.ANN.models.test_models import PortiConvAtt
from portiloop_software.portiloop_python.ANN.utils import LoggerWandb, get_configs, set_seeds
from portiloop_software.portiloop_python.ANN.data.moda_data import generate_dataloader
from portiloop_software.portiloop_python.ANN.data.mass_data import get_dataloaders_mass, get_dataloaders_sleep_stage
from portiloop_software.portiloop_python.ANN.models.test_models import PortiResNet
from torchsummary import summary
from portiloop_software.portiloop_python.ANN.models.lstm import PortiloopNetwork

print("Setting up config...")
experiment_name = 'OriginalModel_SleepStage1'
test_set = True
seed = 42
set_seeds(seed)
config_dict = get_configs(experiment_name, test_set, seed)
print("Done...")

print("Loading data...")
config_dict['batch_size_validation'] = 4096
# config_dict['seq_len'] = 1
# config_dict['batch_size'] = 64
# config_dict['window_size'] = 250
# train_loader, val_loader, batch_size_validation, _, _, _ = generate_dataloader(config_dict)
# train_loader, val_loader = get_dataloaders_mass(config_dict)
# print(next(iter(val_loader))[0].shape)

train_loader, val_loader = get_dataloaders_sleep_stage(config_dict)

print("Done...")

print("Initializing model...")
model = PortiloopNetwork(config_dict)
device = config_dict["device_train"]
model.to(device)
summary(model)

# depth = 5
# model = PortiResNet(depth, config_dict['hidden_size'], config_dict['nb_rnn_layers'])
# summary(model)
# model.to(config_dict["device_train"])
print("Done...")

print("Starting training...")
recurrent = True
save_model = True
unique_name = False
config_dict['lr_adam'] = 0.0005

print("Starting logger...")
wandb_project = f"full-dataset-public"
wandb_group = 'DEFAULT'
logger = LoggerWandb(experiment_name, config_dict, wandb_project, group=wandb_group)
print("Done...")

train(
    train_loader, 
    val_loader, 
    model, 
    recurrent, 
    logger, 
    save_model, 
    unique_name, 
    config_dict
    )
print("Done...")
