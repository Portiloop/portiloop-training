from copy import deepcopy
from random import shuffle

import torch

from Software.python.Utils.utils import sample_config_dict
from pareto_search import SurrogateModel, META_MODEL_DEVICE, train_surrogate, load_network_files, transform_config_dict_to_input

finished_experiment, _ = load_network_files()
shuffle(finished_experiment)
test_network = finished_experiment.pop(0)
test_network2 = finished_experiment.pop(0)
test_network3 = finished_experiment.pop(0)
test_network4 = finished_experiment.pop(0)
meta_model = SurrogateModel()
meta_model.to(META_MODEL_DEVICE)
meta_model.train()
meta_model, meta_loss = train_surrogate(meta_model, deepcopy(finished_experiment))
print(f"DEBUG: meta_loss = {meta_loss}")
meta_model.eval()
test = test_network
print(test)
with torch.no_grad():
    config_dict = test["config_dict"]
    x = transform_config_dict_to_input(config_dict)
    test_model_loss = meta_model(x).item()
print(f"DEBUG: test model loss = {test_model_loss} / real loss = {test['cost_software']}")
test = test_network2
print(test)
with torch.no_grad():
    config_dict = test["config_dict"]
    x = transform_config_dict_to_input(config_dict)
    test_model_loss = meta_model(x).item()
print(f"DEBUG: test model loss = {test_model_loss} / real loss = {test['cost_software']}")
test = test_network3
print(test)
with torch.no_grad():
    config_dict = test["config_dict"]
    x = transform_config_dict_to_input(config_dict)
    test_model_loss = meta_model(x).item()
print(f"DEBUG: test model loss = {test_model_loss} / real loss = {test['cost_software']}")
test = test_network4
print(test)
with torch.no_grad():
    config_dict = test["config_dict"]
    x = transform_config_dict_to_input(config_dict)
    test_model_loss = meta_model(x).item()
print(f"DEBUG: test model loss = {test_model_loss} / real loss = {test['cost_software']}")
test, _ = sample_config_dict(name="test2", previous_exp={}, all_exp=finished_experiment)
#print(test)
with torch.no_grad():
    config_dict = test
    x = transform_config_dict_to_input(config_dict)
    test_model_loss = meta_model(x).item()
print(f"DEBUG: test model loss = {test_model_loss}")
test, _ = sample_config_dict(name="test2", previous_exp={}, all_exp=finished_experiment)
#print(test)
with torch.no_grad():
    config_dict = test
    x = transform_config_dict_to_input(config_dict)
    test_model_loss = meta_model(x).item()
print(f"DEBUG: test model loss = {test_model_loss}")
test, _ = sample_config_dict(name="test2", previous_exp={}, all_exp=finished_experiment)
#print(test)
with torch.no_grad():
    config_dict = test
    x = transform_config_dict_to_input(config_dict)
    test_model_loss = meta_model(x).item()
print(f"DEBUG: test model loss = {test_model_loss}")
