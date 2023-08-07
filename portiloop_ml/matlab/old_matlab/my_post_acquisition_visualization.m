path = "../dataset/13042021_night_lp35hz/";

signal_lp = load(path+"eeg_0_filtered_lp_35_1.txt");
signal_raw = downsample(load(path+"eeg_0_data_1.txt"), 2);

fe = 250;
tot_time = size(signal_lp, 1)/fe;
size_signal = size(signal_lp,1);
time_vect = linspace(0,size_signal/fe, size_signal);

signal_raw_bp = normalize(bandpass(signal_raw, [0.5 35], fe));

figure
plot(time_vect, signal_raw_bp);
axis([0 30 -20 20]);
figure
plot(time_vect, signal_lp);
axis([0 30 -20 20]);