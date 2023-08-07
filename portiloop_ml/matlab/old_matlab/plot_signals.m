data = importdata("eeg_0_data_0(1).txt");

data2 = importdata("eeg_0_filtered_0(1).txt");
filterOnline = out.onlineFilter;
filterOffline = out.offlineFilter;
raw_data = out.data1;
time_vect = out.tout;
figure
hold on
subplot(3,1,1);
plot(time_vect, raw_data);
axis([0 30 0.0055-750e-5 0.0055+750e-5]);
ylabel("Amplitude (V)");
title("Raw signal");
xlabel("Time (s)");
subplot(3,1,2);
plot(time_vect, filterOffline);
axis([0 30 -450e-6 450e-6]);
ylabel("Amplitude (V)");
xlabel("Time (s)");
title("Filtered signal (offline) with a bandpass 12-16 Hz");
subplot(3,1,3);
plot(time_vect, filterOnline);
axis([0 30 -450e-6 450e-6]);
ylabel("Amplitude (V)");
xlabel("Time (s)");
title("Filtered signal (online) with a bandpass 12-16 Hz");
