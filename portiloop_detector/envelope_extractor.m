%% load data
path = "../dataset/";
dataset_small = load(path+"dataset_small_250_matlab.txt");

%% 

signal = dataset_small(:,1);
size_signal = size(signal,1);
fe = 250;
spindles_simulink = dataset_small(:,2) == 1;
spindles = dataset_small(:,2) == 1;
duration = size_signal/fe;
time_vect = linspace(0,size_signal/fe, size_signal);
signal_simulink = [time_vect' signal];
figure
hold on
i = 0;
while i < length(spindles)-1
    i = i+1;
    idx = i;
    while i < length(spindles)-1 && spindles(i+1) == spindles(idx)
       i = i + 1; 
    end
    c = 'b';
    if spindles(idx)
       c = 'r'; 
    end
    plot(time_vect(idx:i), signal(idx:i), 'Color', c);
end
axis([140 160 -20 20]);

%% filtered signal

signal_filtered_simulink = out.signal_filtered(10:end);
time_vect_simulink = out.tout(10:end);
signal_filtered = bandpass(signal, [9, 16], fe);
signal_filtered_simulink = normalize(signal_filtered_simulink);
signal_filtered = normalize(signal_filtered);
figure(2)
hold on
i = 0;
while i < length(signal_filtered_simulink)-1
    i = i+1;
    idx = i;
    while i < length(signal_filtered_simulink)-1 && spindles_simulink(i+1) == spindles_simulink(idx)
       i = i + 1; 
    end
    c = 'b';
    if spindles_simulink(idx)
       c = 'r'; 
    end
    plot(time_vect_simulink(idx:i), signal_filtered_simulink(idx:i), 'Color', c);
end

axis([140 160 -5 5]);
figure(3)
hold on
i = 0;
while i < length(signal_filtered)-1
    i = i+1;
    idx = i;
    while i < length(signal_filtered)-1 && spindles(i+1) == spindles(idx)
       i = i + 1; 
    end
    c = 'b';
    if spindles(idx)
       c = 'r'; 
    end
    plot(time_vect(idx:i), signal_filtered(idx:i), 'Color', c);
end
axis([140 160 -5 5]);

%% envelope extraction
figure(6)
hold on
[envelope_hilbert_simulink, ~] = envelope(signal_filtered_simulink);
i = 0;
while i < length(envelope_hilbert_simulink)-1
    i = i+1;
    idx = i;
    while i < length(envelope_hilbert_simulink)-1 && spindles_simulink(i+1) == spindles_simulink(idx)
       i = i + 1; 
    end
    c = 'b';
    if spindles_simulink(idx)
       c = 'r'; 
    end
    plot(time_vect_simulink(idx:i), envelope_hilbert_simulink(idx:i), 'Color', c);
end
axis([140 160 -5 5]);
figure(7)
hold on
[envelope_hilbert, ~] = envelope(signal_filtered);
i = 0;
while i < length(envelope_hilbert)-1
    i = i+1;
    idx = i;
    while i < length(envelope_hilbert)-1 && spindles(i+1) == spindles(idx)
       i = i + 1; 
    end
    c = 'b';
    if spindles(idx)
       c = 'r'; 
    end
    plot(time_vect(idx:i), envelope_hilbert(idx:i), 'Color', c);
end
axis([140 160 -5 5]);

%% homemade envelope
figure(2)
hold on
envelope_homemade_simulink = abs(signal_filtered_simulink);

moving_average = envelope_homemade_simulink(1);
alpha = 0.01;
for i=1:size(envelope_homemade_simulink,1)
    delta = envelope_homemade_simulink(i) - moving_average;
    moving_average = moving_average + alpha*delta;
    envelope_homemade_simulink(i) = moving_average;
end
i = 0;
while i < length(envelope_homemade_simulink)-1
    i = i+1;
    idx = i;
    while i < length(envelope_homemade_simulink)-1 && spindles_simulink(i+1) == spindles_simulink(idx)
       i = i + 1; 
    end
    c = 'yellow';
    if spindles_simulink(idx)
       c = 'magenta'; 
    end
    plot(time_vect_simulink(idx:i), envelope_homemade_simulink(idx:i), 'Color', c);
end
axis([140 160 -5 5]);
% figure(5)
% hold on
% envelope_shannon = ShannonEnergy(signal_filtered);
% envelope_shannon = normalize(envelope_shannon);
% plot(time_vect, envelope_shannon);
% plot(time_vect(spindles), envelope_shannon(spindles));
% axis([140 160 -5 5]);

%% shannon energy envelope

figure(10);
hold on
pas = 70;
envelope_valentin = signal_filtered;
for i=pas+1:length(signal_filtered)
    envelope_valentin(i) = max(signal_filtered(i-pas:i));
end
plot(time_vect, envelope_valentin);
plot(time_vect(spindles), envelope_valentin(spindles));

%% save
output_envelope = [dataset(:,1), envelope_hilbert_simulink, dataset(:,2)];

writematrix(output_envelope, path+"dataset_small_envelope_matlab.txt");

%% fait maison

signal_filtered_simulink = out.signal_filtered;
time_vect_simulink = out.tout;
signal_filtered_simulink = normalize(signal_filtered_simulink);
figure(3)
hold on
plot(time_vect_simulink, signal_filtered_simulink);
plot(time_vect_simulink(spindles), signal_filtered_simulink(spindles));
axis([140 160 -5 5]);


envelope_homemade = out.envelope_homemade.Data(1,:);
plot(out.envelope_homemade.Time, envelope_homemade);
plot(out.envelope_homemade.Time(spindles), envelope_homemade(spindles));

