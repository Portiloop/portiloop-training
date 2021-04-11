%% load data from portiloop

path = "../dataset/";
fe = 250;
n_files = 14;
tot_time_file = fe*30*60;
tot_time = tot_time_file*n_files;
signal = zeros(tot_time, 1);
for i=0:13
    signal(i*tot_time_file+1:(i+1)*tot_time_file) = downsample(load(path+"data_07072020_" + i + ".txt"),2);
end

writematrix(signal, path+"0707_portiloop_dataset_250.txt");
