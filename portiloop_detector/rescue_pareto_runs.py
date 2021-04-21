from pareto_search import *

all_experiments, pareto_front = load_files()
for i in range(len(all_experiments)):
    all_experiments[i]["cost_hardware"] *= MAX_NB_PARAMETERS
    all_experiments[i]["cost_software"] *= MAX_LOSS

for i in range(len(pareto_front)):
    pareto_front[i]["cost_hardware"] *= MAX_NB_PARAMETERS
    pareto_front[i]["cost_software"] *= MAX_LOSS
dump_files(all_experiments, pareto_front)
