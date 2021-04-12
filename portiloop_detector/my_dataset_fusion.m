%% load data from portiloop

path = "../dataset/";
fe = 250;
n_files = 9;
tot_time_file = fe*30*60;
tot_time = tot_time_file*n_files;
signal = zeros(tot_time, 1);
for i=0:(n_files-1)
    k = i +1;
    signal(i*tot_time_file+1:(i+1)*tot_time_file) = downsample(load(path+"elec_0_data_09082020_" + k),2);
end

writematrix(signal, path+"0908_portiloop_dataset_250.txt");
