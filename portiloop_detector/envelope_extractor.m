%% load data
path = "../dataset/";
dataset = load(path+"dataset_small_250_matlab.txt");

%% 

signal = dataset(:,1);
size_signal = size(signal,1);
fe = 250;
spindles = dataset(:,2) == 1;
duration = size_signal/fe;
time_vect = linspace(0,size_signal/fe, size_signal);
signal_simulink = [time_vect' signal];
figure(1)
hold on
plot(time_vect, signal);
plot(time_vect(spindles), signal(spindles));
axis([140 160 -5 5]);

%% filtered signal

%signal_filtered = out.signal_filtered;
%time_vect_simulink = out.tout;
signal_filtered = bandpass(signal, [9, 16], fe);
signal_filtered = normalize(signal_filtered);
figure(2)
hold on
plot(time_vect, signal_filtered);
plot(time_vect(spindles), signal_filtered(spindles));
axis([140 160 -5 5]);

%% envelope extraction

[envelope_hilbert, aaa] = envelope(signal_filtered);
plot(time_vect, envelope_hilbert);
plot(time_vect(spindles), envelope_hilbert(spindles));


%% save
output_envelope = [dataset(:,1), envelope_hilbert, dataset(:,2)];

writematrix(output_envelope, path+"dataset_small_envelope_matlab.txt");

%% fait maison

signal_filtered = out.signal_filtered;
time_vect_simulink = out.tout;
signal_filtered = normalize(signal_filtered);
figure(3)
hold on
plot(time_vect_simulink, signal_filtered);
plot(time_vect_simulink(spindles), signal_filtered(spindles));
axis([140 160 -5 5]);


envelope_homemade = out.envelope_homemade.Data(1,:);
plot(out.envelope_homemade.Time, envelope_homemade);
plot(out.envelope_homemade.Time(spindles), envelope_homemade(spindles));

