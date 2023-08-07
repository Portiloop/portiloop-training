from random import seed

from pareto_search import sample_config_dict

# finished_experiment, _ = load_network_files()
#
#
# def from_config_dict_to_vector(config_dict):
#     return np.array([float(config_dict["seq_len"]),  # idk why, but needed
#      config_dict["nb_channel"],
#      config_dict["hidden_size"],
#      int(config_dict["seq_stride_s"] * config_dict["fe"]),
#      config_dict["nb_rnn_layers"],
#      int(config_dict["window_size_s"] * config_dict["fe"]),
#      config_dict["nb_conv_layers"],
#      config_dict["stride_pool"],
#      config_dict["stride_conv"],
#      config_dict["kernel_conv"],
#      config_dict["kernel_pool"],
#      config_dict["dilation_conv"],
#      config_dict["dilation_pool"]])
#
#
# features = np.array([from_config_dict_to_vector(exp['config_dict']) for exp in finished_experiment])
# labels = np.array([exp['cost_software'] for exp in finished_experiment])
# train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=0)
# forest = LinearRegression()
#
# print(forest.fit(train_features, train_labels))
# print(forest.score(train_features, train_labels))
# print(forest.score(test_features, test_labels))
#
# pred = forest.predict(test_features)
# print(pred)
# print(test_labels)
# loss = sklearn.metrics.mean_squared_error(test_labels, pred, sample_weight=None, multioutput='uniform_average', squared=True)
# print(loss)

seed(0)
config_dict = sample_config_dict(f"variance_test_{0}", previous_exp={}, all_exp=[])
print(config_dict)
seed(0)
config_dict = sample_config_dict(f"variance_test_{0}", previous_exp={}, all_exp=[])
print(config_dict)
seed(0)
config_dict = sample_config_dict(f"variance_test_{0}", previous_exp={}, all_exp=[])
print(config_dict)
