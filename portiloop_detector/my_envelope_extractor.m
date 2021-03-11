%% load data
path = "../dataset/";
dataset = load(path+"dataset_big_envelope_fusion_pf.txt");

%% 

signal = dataset(:,1);
size_signal = size(signal,1);
tot_time =size_signal/fe;

fe = 250;
% spindles_simulink = dataset(:,2) == 1;
% spindles = dataset(:,2) == 1;
duration = size_signal/fe;
time_vect = linspace(0,size_signal/fe, size_signal);
signal_simulink = [time_vect' signal];
% figure
% hold on
% i = 0;
% while i < length(spindles)-1
%     i = i+1;
%     idx = i;
%     while i < length(spindles)-1 && spindles(i+1) == spindles(idx)
%        i = i + 1; 
%     end
%     c = 'b';
%     if spindles(idx)
%        c = 'r'; 
%     end
%     plot(time_vect(idx:i), signal(idx:i), 'Color', c);
% end
% plot(time_vect, signal);
% plot(time_vect(spindles), signal(spindles));
% axis([140 160 -5 5]);

%% filtered signal

signal_filtered_simulink = out.signal_filtered(10:end);
time_vect_simulink = out.tout(10:end);
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
% figure(2)
% hold on
% i = 0;
% while i < length(signal_filtered_simulink)-1
%     i = i+1;
%     idx = i;
%     while i < length(signal_filtered_simulink)-1 && spindles_simulink(i+1) == spindles_simulink(idx)
%        i = i + 1; 
%     end
%     c = 'b';
%     if spindles_simulink(idx)
%        c = 'r'; 
%     end
%     plot(time_vect_simulink(idx:i), signal_filtered_simulink(idx:i), 'Color', c);
% end

%% homemade envelope
envelope_homemade_simulink = signal_filtered_simulink.^2;

moving_average = envelope_homemade_simulink(1);
alpha = 0.01;
for i=1:size(envelope_homemade_simulink,1)
    delta = envelope_homemade_simulink(i) - moving_average;
    moving_average = moving_average + alpha*delta;
    envelope_homemade_simulink(i) = moving_average;
end

% figure(2)
% hold on
% i = 0;
% while i < length(envelope_homemade_simulink)-1
%     i = i+1;
%     idx = i;
%     while i < length(envelope_homemade_simulink)-1 && spindles_simulink(i+1) == spindles_simulink(idx)
%        i = i + 1; 
%     end
%     c = 'green';
%     if spindles_simulink(idx)
%        c = 'magenta'; 
%     end
%     plot(time_vect_simulink(idx:i), envelope_homemade_simulink(idx:i), 'Color', c);
% end
% axis([140 160 -20 20]);

%% save
dataset_final = load(path+"dataset_big_fusion_standardized.txt");

output_envelope = single([dataset_final(:,1), envelope_homemade_simulink, dataset_final(:,2)]);

writematrix(output_envelope, path+"dataset_big_fusion_standardized_envelope.txt");
