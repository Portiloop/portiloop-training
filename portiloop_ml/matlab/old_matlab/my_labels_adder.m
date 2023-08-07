dataset_final = load(path+"13042021_portiloop_dataset_250_standardized_envelope_pf.txt");
spindles = load(path+"detVect_13042021_portiloop_dataset_250_standardized.txt");

%%
datasetupdate = single([dataset_final(:,1), dataset_final(:,2), dataset_final(:, 3), spindles]);

writematrix(datasetupdate, path+"13042021_portiloop_dataset_250_standardized_envelope_pf_labeled.txt");
