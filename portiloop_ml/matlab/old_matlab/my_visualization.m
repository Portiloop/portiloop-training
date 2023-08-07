path = "../dataset/";
phase = 'p1';
size_data = 'big';
dataset = load(path+"dataset_classification_"+phase+"_"+size_data+"_250_matlab_standardized_envelope_pf.txt");
signal = dataset(:,1);
spindles_gs = dataset(:,4) == 1;

fe = 250;
tot_time = size(dataset, 1)/fe;
size_signal = size(signal,1);
time_vect = linspace(0,size_signal/fe, size_signal);

%%
figure
% subplot(2, 1, 1)
hold on
decallage = (145*70+15)*fe;
i = 0;
plot_data = signal(decallage:decallage + 115*fe);
plot_spindles = spindles_gs(decallage:decallage + 115*fe);
while i < length(plot_data)-1
    i = i+1;
    idx = i;
    while i < length(plot_data)-1 && plot_spindles(i+1) == plot_spindles(idx)
       i = i + 1; 
    end
    c = 'b';
    if plot_spindles(idx)
       c = 'r'; 
    end
    plot(time_vect(idx:i+1), plot_data(idx:i+1), 'Color', c);
end
axis([0, 10, -10, 10]);