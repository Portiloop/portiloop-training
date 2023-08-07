%% load data from portiloop

path = "../dataset/13042021_night_lp35hz/";
fe = 250;
n_files = 3;
tot_time_file = fe*30*60;
tot_time = tot_time_file*n_files;
signal = zeros(tot_time, 1);
envelope = zeros(tot_time, 1);
for i=0:(n_files-1)
    k = i +1;
    envelope(i*tot_time_file+1:(i+1)*tot_time_file) = load(path+"eeg_0_filtered_12_16_" + k + ".txt");
    signal(i*tot_time_file+1:(i+1)*tot_time_file) = load(path+"eeg_0_filtered_lp_35_" + k + ".txt");
end

writematrix(signal, path+"13042021_portiloop_dataset_250_standardized.txt");
writematrix(envelope, path+"13042021_portiloop_dataset_250_envelope.txt");
dataset = [signal, envelope];
writematrix(dataset, path+"13042021_portiloop_dataset_250_standardized_envelope.txt");
