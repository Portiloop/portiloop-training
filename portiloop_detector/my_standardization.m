%% load data
path = "../dataset/";
dataset = load(path+"dataset_big_envelope_fusion_pf.txt");

%% load data from portiloop

path = "../dataset/";
dataset = load(path+"0908_portiloop_dataset_250.txt");
dataset_full = load(path+"0908_portiloop_dataset_250_standardized_envelope_pf_labeled.txt");
dataset = [dataset, dataset_full(:, 2:end)];
%% 
signal = dataset(:,1);
spindles_gs = dataset(:,4) == 1;
spindles_hugo = dataset(:,4) == 0.8;
fe = 250;
tot_time = size(dataset, 1)/fe;
size_signal = size(dataset, 1);
time_vect = linspace(0,size_signal/fe, size_signal);

%%
out = sim('filter_lp',tot_time);
sim_filtered_lp = [out.filtered_simulink(10:end); out.filtered_simulink(end-7:end)];
time_vect = out.tout(2:end);

%%
%sim_filtered_lp = lowpass(signal, 30, fe);
% % wo = 60/(250/2);  
% % bw = wo/35;
% % [b,a] = iirnotch(wo,bw);
% % signal_filt = filtfilt(b, a, signal);
% % sim_filtered_bp_notch = bandpass(signal_filt, [0.3 30], fe);
sim_filtered_lp = bandpass(signal, [0.5 35], fe);
% time_vect = linspace(0, tot_time, tot_time*fe);
% % figure
% % hold on
% % plot(time_vect, signal);
% % plot(time_vect, signal_filt);
% % plot(time_vect, sim_filtered_bp_notch);
% % axis([580 610 -2e-4 2e-4]);
% 
% % norm_sim_filtered_bp = normalize(sim_filtered_bp);
% % norm_sim_filtered_bp_notch = normalize(sim_filtered_bp_notch);
% % figure
% % hold on
% % plot(time_vect, norm_sim_filtered_bp);
% % plot(time_vect, norm_sim_filtered_bp_notch);
% % axis([580 610 -5 5]);

%%
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
% sim_filtered_bp = normalize(sim_filtered_bp);

plot_data = lp_standard(1:1800*fe);
figure
% subplot(2, 1, 1)
hold on
i = 0;
while i < length(plot_data)-1
    i = i+1;
    idx = i;
    while i < length(plot_data)-1 && spindles_gs(i+1) == spindles_gs(idx) && spindles_hugo(i+1) == spindles_hugo(idx)
       i = i + 1; 
    end
    c = 'b';
    if spindles_gs(idx)
       c = 'r'; 
    end
    if spindles_hugo(idx)
       c = 'm'; 
    end
    plot(time_vect(idx:i), plot_data(idx:i), 'Color', c);
end
axis([0 30 -20 20]);
title("lp");
% plot_data = sim_filtered_bp(1:200*fe);
% subplot(2, 1, 2)
% hold on
% i = 0;
% while i < length(plot_data)-1
%     i = i+1;
%     idx = i;
%     while i < length(plot_data)-1 && spindles_gs(i+1) == spindles_gs(idx) && spindles_hugo(i+1) == spindles_hugo(idx)
%        i = i + 1; 
%     end
%     c = 'b';
%     if spindles_gs(idx)
%        c = 'r'; 
%     end
%     if spindles_hugo(idx)
%        c = 'm'; 
%     end
%     plot(time_vect(idx:i), plot_data(idx:i), 'Color', c);
% end
% axis([140 170 -5 5]);
% title("bp");
%% fft

mydata=lp_standard;
mysrate=fe;
mynpnts=size(sim_filtered_lp, 1);
%mydata_filtered=highpass(mydata,0.1,mysrate);
%mydata=mydata_filtered;mynpnts = length(mydata);
mytime  = (0:mynpnts-1)/mysrate;mydata_demean=mydata-mean(mydata);% plot the time-domain signal
figure
plot(mytime,mydata_demean)
xlabel('Time (s)'), ylabel('Voltage ')
zoom on% static spectral analysis
myhz = linspace(0,mysrate/2,floor(mynpnts/2)+1);
myampl = 2*abs(fft(mydata_demean)/mynpnts);
mypowr = myampl.^2;
figure
hold on
plot(myhz,myampl(1:length(myhz)),'k','linew',2)
plot(myhz,mypowr(1:length(myhz)),'r','linew',2)


%% save

%output_signal = single(lp_standard(:,1));%, dataset(1:end-8,4)]);
output_signal = single([lp_standard(:,1), dataset(:, 2:end)]);

writematrix(output_signal, path+"0908_portiloop_dataset_250_standardized_simulink_envelope_pf_labeled.txt");

%%
plot(time_vect, output_signal);