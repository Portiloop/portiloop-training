%% load data
path = "../dataset/";
dataset = load(path+"dataset_big_envelope_fusion_pf.txt");

%%
signal = dataset(:,1);
spindles_gs = dataset(:,4) == 1;
spindles_hugo = dataset(:,4) == 0.8;
fe = 250;
tot_time = size(dataset, 1)/fe;

%%
sim_filtered = out.filtered_simulink(10:end);
time_vect = out.tout(10:end);

%%
moving_average = sim_filtered(1);
moving_variance = 0;
alpha = 0.001;
for i=2:size(sim_filtered,1)
    delta = sim_filtered(i) - moving_average;
    moving_average = moving_average + alpha*delta;
    moving_variance = (1-alpha)*(moving_variance + alpha*delta.^2);
    moving_std = sqrt(moving_variance);
    sim_filtered(i) = (sim_filtered(i) - moving_average)./moving_std;
end
plot_data = sim_filtered(1:200*fe);
figure
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
       c = 'g'; 
    end
    plot(time_vect(idx:i), plot_data(idx:i), 'Color', c);
end
axis([140 160 -20 20]);

%% save

output_signal = single([sim_filtered(:,1), dataset(1:end-8,4)]);

writematrix(output_signal, path+"dataset_big_fusion_standardized.txt");
