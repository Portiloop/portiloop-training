from matplotlib import pyplot as plt

from pareto_search import load_network_files, path_dataset, PARETO_ID
from utils import MAX_NB_PARAMETERS

finished_experiment, pareto_front = load_network_files()

best = pareto_front[0]
print(f"Best network with {best['cost_hardware']} params and a loss validation of {best['cost_software']} : {best['config_dict']}")
#
# plt.hist([exp["cost_hardware"] for exp in finished_experiment], bins=100, range=(0, MAX_NB_PARAMETERS))
# plt.title("histogram")
# plt.show()
#
# implemented_run = [run for run in finished_experiment if "35" in run['config_dict']['experiment_name']]
# for run in implemented_run:
#     print(run['cost_software'])
#     print(run['cost_hardware'])
#     print(run['config_dict'])

plt.clf()
fig = plt.figure()

x_axis = [exp["cost_hardware"] for exp in finished_experiment[:-1]]
y_axis = [exp["cost_software"] for exp in finished_experiment[:-1]]
plt.scatter(x_axis, y_axis, s=2, c='k')
# pareto:
x_axis = [exp["cost_hardware"] for exp in pareto_front]
y_axis = [exp["cost_software"] for exp in pareto_front]
plt.scatter(x_axis, y_axis, s=3, c='r')
plt.plot(x_axis, y_axis, 'r-', linewidth=1)

plt.xlabel(f"Hardware cost")
plt.ylabel(f"Software cost")
# plt.ylim(top=0.1)

fig.set_size_inches(7,3)
fig.subplots_adjust(bottom=0.15)
fig.savefig(path_dataset / f"pareto_plot_{PARETO_ID}.pdf", dpi=200)

plt.show()
