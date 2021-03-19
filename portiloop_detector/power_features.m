%% load
path = "../dataset/";
dataset = load(path+"dataset_test_fusion_standardized_envelope.txt");
output_envelope = single(dataset);
%% data selection
Fs = 250;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 125;             % Length of signal
t = (0:L-1)*T;        % Time vector

f = Fs*(0:(L/2))/L;
f28 = find(2 <= f & f<=8);
f916 = find(9 <= f & f<=16);
signal = output_envelope(:,1);
size_signal = size(signal,1);

%% compute
r = zeros(size_signal,1);
delay = 0;
length = size_signal;
for i=L:length
    r(i) = compute_ratio(i+delay, signal, L, f28, f916);
end

%% save 

datasetupdate = single([output_envelope(:,1), output_envelope(:,2), r, output_envelope(:,3)]);

writematrix(datasetupdate, path+"dataset_test_fusion_standardized_envelope_pf.txt");

%% plot

signal = dataset(delay:delay+length-1,1);
size_signal = size(signal,1);
fe = 250;
spindles = dataset(delay:delay+length-1,2) == 1;
duration = size_signal/fe;
time_vect = linspace(0,size_signal/fe, size_signal);
signal_simulink = [time_vect' signal];
figure
hold on
plot(time_vect, signal);
plot(time_vect(spindles), signal(spindles));
plot(time_vect, r);
axis([0 30 -5 5]);