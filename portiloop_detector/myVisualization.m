dataset = load(path+"0908_portiloop_dataset_250_standardized_envelope_pf_labeled.txt");
signal = dataset(:,1);
spindles_gs = dataset(:,4) == 1;
spindles_hugo = dataset(:,4) == 0.8;
fe = 250;
tot_time = size(dataset, 1)/fe;
size_signal = size(signal,1);
time_vect = linspace(0,size_signal/fe, size_signal);

%%
figure
% subplot(2, 1, 1)
hold on
i = 0;
plot_data = signal(1:1800*fe);
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
