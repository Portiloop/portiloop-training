import copy

import numpy as np

from pareto_search import *

all_experiments, _ = load_network_files()
pareto_front = []
new_all_exp = []
# for i in range(len(all_experiments)):
#     all_experiments[i]["cost_hardware"] *= MAX_NB_PARAMETERS
#     all_experiments[i]["cost_software"] *= MAX_LOSS
#
# for i in range(len(pareto_front)):
#     pareto_front[i]["cost_hardware"] *= MAX_NB_PARAMETERS
#     pareto_front[i]["cost_software"] *= MAX_LOSS

while len(all_experiments) > 0:
    exp = all_experiments.pop()
    same_configs = [exp]
    all_exps_copy = copy.deepcopy(all_experiments)
    for e in all_exps_copy:
        if same_config_dict(e['config_dict'], same_configs[0]['config_dict']):
            same_configs.append(e)
            all_experiments.remove(e)
            print('same dict found')
    exp['cost_software'] = min([e['cost_software'] for e in same_configs])
    new_all_exp.append(exp)
    pareto_front = update_pareto(exp, pareto_front)

dump_network_files(new_all_exp, pareto_front)
