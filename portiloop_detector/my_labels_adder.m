dataset_final = load(path+"0908_portiloop_dataset_250_standardized_envelope_pf.txt");
spindles = load(path+"detVect_0908_portiloop_dataset_250_standardized.txt");

%%
datasetupdate = single([dataset_final(:,1), dataset_final(:,2), dataset_final(:, 3), spindles]);

writematrix(datasetupdate, path+"0908_portiloop_dataset_250_standardized_envelope_pf_labeled.txt");
