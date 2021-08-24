%% extract data
path = "../dataset/";

raw_default = load(path + "13042021_night_lp35hz/eeg_0_data_3.txt");
fe = 250;
default_fe = 500;
raw = downsample(raw_default, default_fe/fe);
tot_time = size(raw,1)/fe;
size_signal = size(raw, 1);
time_vect = linspace(0,size_signal/fe, size_signal);

%% plot raw signal

f = figure;
plot(time_vect, raw);
axis([80 110 -0.0105-350e-5 -0.0105+350e-5]);
ylabel("Amplitude (V)");
title("Signal brut");
xlabel("Temps (s)");
set(gca,'color','white')

set(gcf,'color','white')
f.Position = [10 0 1500 500];




%% plot lp filtered signal

signal = raw;
out = sim('filter_lp_with_no_notch',tot_time);
sim_filtered_lp_no_notch = [out.filtered_simulink(10:end); out.filtered_simulink(end-7:end)];
time_vect = out.tout(2:end);


f = figure;
f.Position = [10 0 1500 500];

plot(time_vect, sim_filtered_lp_no_notch);
axis([80 110 -0.01075-150e-6 -0.01075+150e-6]);
ylabel("Amplitude (V)");
title("Signal filtré avec un passe-bas");
xlabel("Temps (s)");
set(gca,'color','white')

set(gcf,'color','white')


% 
% 
% mydata=sim_filtered_lp_no_notch;
% mysrate=fe;
% mynpnts=size(sim_filtered_lp_no_notch, 1);
% %mydata_filtered=highpass(mydata,0.1,mysrate);
% %mydata=mydata_filtered;mynpnts = length(mydata);
% mytime  = (0:mynpnts-1)/mysrate;mydata_demean=mydata-mean(mydata);% plot the time-domain signal
% figure
% plot(mytime,mydata_demean)
% xlabel('Time (s)'), ylabel('Voltage ')
% zoom on% static spectral analysis
% myhz = linspace(0,mysrate/2,floor(mynpnts/2)+1);
% myampl = 2*abs(fft(mydata_demean)/mynpnts);
% mypowr = myampl.^2;
% figure
% hold on
% plot(myhz,myampl(1:length(myhz)),'k','linew',2)
% plot(myhz,mypowr(1:length(myhz)),'r','linew',2)


%% plot lp + notch filtered signal

signal = raw;

out = sim('filter_lp',tot_time);
sim_filtered_lp = [out.filtered_simulink(10:end); out.filtered_simulink(end-7:end)];
time_vect = out.tout(2:end);


f = figure;
f.Position = [10 0 1500 500];
plot(time_vect, sim_filtered_lp);
axis([80 110 -0.01075-150e-6 -0.01075+150e-6]);
ylabel("Amplitude (V)");
title("Signal filtré avec un passe-bas et un coupe-bande");
xlabel("Temps (s)");
set(gca,'color','white')

set(gcf,'color','white')


% mydata=sim_filtered_lp;
% mysrate=fe;
% mynpnts=size(sim_filtered_lp, 1);
% %mydata_filtered=highpass(mydata,0.1,mysrate);
% %mydata=mydata_filtered;mynpnts = length(mydata);
% mytime  = (0:mynpnts-1)/mysrate;mydata_demean=mydata-mean(mydata);% plot the time-domain signal
% figure
% plot(mytime,mydata_demean)
% xlabel('Time (s)'), ylabel('Voltage ')
% zoom on% static spectral analysis
% myhz = linspace(0,mysrate/2,floor(mynpnts/2)+1);
% myampl = 2*abs(fft(mydata_demean)/mynpnts);
% mypowr = myampl.^2;
% figure
% hold on
% plot(myhz,myampl(1:length(myhz)),'k','linew',2)
% plot(myhz,mypowr(1:length(myhz)),'r','linew',2)


%% plot lp + notch filtered and standardized signal

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



f = figure;
f.Position = [10 0 800 300];

plot(time_vect, lp_standard);
axis([146 156 -10 10]);
ylabel("Amplitude (V)");
title("Signal filtré et standardizé");
xlabel("Temps (s)");
set(gca,'color','white')

set(gcf,'color','white')


%% same plot with MASS data

mass = load(path + "dataset_classification_p1_big_250_matlab_standardized_envelope_pf.txt");

%% plot 
signal_mass = mass(:, 1);

tot_time = size(signal_mass,1)/fe;
size_signal = size(signal_mass, 1);
time_vect_mass = linspace(0,size_signal/fe, size_signal);

f = figure;
f.Position = [10 0 800 300];

plot(time_vect_mass, signal_mass);
axis([146 156 -10 10]);
ylabel("Amplitude (V)");
title("Signal filtré et standardizé");
xlabel("Temps (s)");
set(gca,'color','white')

set(gcf,'color','white')
