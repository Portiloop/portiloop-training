from matplotlib import pyplot as plt

from pareto_search import load_network_files, MAX_NB_PARAMETERS

finished_experiment, pareto_front = load_network_files()

best = pareto_front[0]
print(f"Best network with {best['cost_hardware']} params and a loss validation of {best['cost_software']} : {best['config_dict']}")

plt.hist([exp["cost_hardware"] for exp in finished_experiment], bins=100, range=(0, MAX_NB_PARAMETERS))
plt.title("histogram")
plt.show()

implemented_run = [run for run in finished_experiment if "284" in run['config_dict']['experiment_name']]
for run in implemented_run:
    print(run['cost_software'])
    print(run['cost_hardware'])
    print(run['config_dict'])