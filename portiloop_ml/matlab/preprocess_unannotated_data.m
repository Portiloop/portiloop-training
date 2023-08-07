path = "../dataset/";
dataset_filename = 'potiloop_sleep_age_young11.txt';
final_dataset_filename = strcat('preprocessed_', dataset_filename);
data_sampling_rate = 100;
raw_data = load(path + dataset_filename);
%% resample
fe = 250;
data_250_matlab = resample(raw_data, fe, data_sampling_rate);
%% generate vector
data_250_matlab = [data_250_matlab];

%% begin standardization
signal = data_250_matlab;
tot_time = length(data_250_matlab)/fe;
%% filter
out = sim('filter_lp',tot_time);
sim_filtered_lp = [out.filtered_simulink(10:end); out.filtered_simulink(end-7:end)];
time_vect = out.tout(2:end);

%% standardize
lp_standard = sim_filtered_lp;
moving_average = lp_standard(1);
moving_variance = 0;
alpha_av = 0.1;
alpha_var = 0.001;
for i=2:size(lp_standard,1)
    delta = lp_standard(i) - moving_average;
    moving_average = moving_average + alpha_av*delta;
    moving_variance = (1-alpha_var)*(moving_variance + alpha_var*delta.^2);
    moving_std = sqrt(moving_variance);
    lp_standard(i) = (lp_standard(i) - moving_average)./moving_std;
end

%% save 

datasetupdate = single([lp_standard]);

writematrix(datasetupdate, path + final_dataset_filename);
