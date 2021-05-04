from pareto_search import load_network_files

finished_experiment, pareto_front = load_network_files()

best = pareto_front[0]
print(f"Best network with {best['cost_hardware']} params and a loss validation of {best['cost_software']} : {best['config_dict']}")

