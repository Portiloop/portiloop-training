path = "../dataset/";
phase = 'p1';
size_data = 'small';
spindle_250 = load(path+"spindles_annotations_classification_"+phase+"_"+size_data+"_at_250hz.txt");
data_256 = load(path+"dataset_"+phase+"_"+size_data+"_at_256_to_resample.txt");
%% resample
fe = 250;
data_250_matlab = resample(data_256, fe, 256);
%% create vector
size_250 = size(spindle_250,1);
%% generate vector
data_250_matlab = [data_250_matlab];
spindle_250 = [spindle_250];
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

%% envelope extraction
time_vect = linspace(0,size_250/fe, size_250);
signal_simulink = [time_vect' signal];
%% filtered signal
out=sim("envelope_extractor_simulink");
signal_filtered_simulink = [out.signal_filtered(10:end); out.signal_filtered(end-7:end)];
time_vect_simulink = out.tout(2:end);
%% standization
moving_average = signal_filtered_simulink(1);
moving_variance = 0;
alpha = 0.001;
for i=2:size(signal_filtered_simulink,1)
    delta = signal_filtered_simulink(i) - moving_average;
    moving_average = moving_average + alpha*delta;
    moving_variance = (1-alpha)*(moving_variance + alpha*delta.^2);
    moving_std = sqrt(moving_variance);
    signal_filtered_simulink(i) = (signal_filtered_simulink(i) - moving_average)./moving_std;
end
%% envelope
envelope_homemade_simulink = signal_filtered_simulink.^2;

moving_average = envelope_homemade_simulink(1);
alpha = 0.01;
for i=1:size(envelope_homemade_simulink,1)
    delta = envelope_homemade_simulink(i) - moving_average;
    moving_average = moving_average + alpha*delta;
    envelope_homemade_simulink(i) = moving_average;
end
%% power features
%% data selection
Fs = fe;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 125;             % Length of signal
t = (0:L-1)*T;        % Time vector

f = Fs*(0:(L/2))/L;
f28 = find(2 <= f & f<=8);
f916 = find(9 <= f & f<=16);
signal = lp_standard;
size_signal = size(signal,1);

%% compute
r = zeros(size_signal,1);
delay = 0;
length = size_signal;
for i=L:length
    r(i) = compute_ratio(i+delay, signal, L, f28, f916);
end

%% save 

datasetupdate = single([lp_standard, envelope_homemade_simulink, r, spindle_250]);

writematrix(datasetupdate, path+"dataset_classification_"+phase + "_"+size_data+"_250_matlab_standardized_envelope_pf.txt");
