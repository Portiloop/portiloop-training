from matplotlib import pyplot as plt

from pareto_search import load_network_files, path_dataset, PARETO_ID, pareto_efficiency

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

fig.set_size_inches(7, 3)
fig.subplots_adjust(bottom=0.15)
fig.savefig(path_dataset / f"pareto_plot_{PARETO_ID}.pdf", dpi=200)

x_axis = pareto_front[-1]["cost_hardware"]
y_axis = pareto_front[-1]["cost_software"]
print(f"lowest cost hardware: {pareto_front[-1]['cost_hardware']}")
pareto_efficiency(pareto_front[-1], finished_experiment)
plt.scatter(x_axis, y_axis, s=50, c='b')
fig.savefig(path_dataset / f"pareto_plot_{PARETO_ID}_lowest_hardware.pdf", dpi=200)


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
fig.set_size_inches(7, 3)
fig.subplots_adjust(bottom=0.15)

x_axis = pareto_front[0]["cost_hardware"]
y_axis = pareto_front[0]["cost_software"]
print(f"lowest cost_software: {pareto_front[0]['cost_software']}")
pareto_efficiency(pareto_front[0], finished_experiment)
plt.scatter(x_axis, y_axis, s=50, c='b')
fig.savefig(path_dataset / f"pareto_plot_{PARETO_ID}_lowest_software.pdf", dpi=200)


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
fig.set_size_inches(7, 3)
fig.subplots_adjust(bottom=0.15)

id = -5
x_axis = pareto_front[id]["cost_hardware"]
y_axis = pareto_front[id]["cost_software"]
print(f"lowest cost_software: {pareto_front[id]['cost_software']}")
pareto_efficiency(pareto_front[id], finished_experiment)
plt.scatter(x_axis, y_axis, s=50, c='b')
fig.savefig(path_dataset / f"pareto_plot_{PARETO_ID}_lowest_software.pdf", dpi=200)

plt.show()
