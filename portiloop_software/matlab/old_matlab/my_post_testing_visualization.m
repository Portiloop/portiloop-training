path = "../Dataset/";
dataset = load(path+"0908_portiloop_dataset_small_250_standardized_simulink_envelope_pf_labeled.txt");
signal = dataset(:,1);
state = load(path + "labels.txt");
state_portiloop = load(path + "labels_portiloop.txt");
fe = 250;
tot_time = size(dataset, 1)/fe;
size_signal = size(signal,1);
time_vect = linspace(0,size_signal/fe, size_signal);

%%
figure
subplot(2, 1, 1)
hold on
seq_stride = 25;
plot_data = signal;
for tp=0:2000%length(state)-1
    idx = tp * seq_stride + 1;
    c = 'b';
    switch state(tp+1)
        case 1
           c = 'g';
        case 2
            c = 'r';
        case 3
            c = 'k';
        case 4
            c = 'b';
    end
    plot(time_vect(idx:idx + seq_stride), plot_data(idx:idx + seq_stride), 'Color', c);
end
axis([120 150 -10 10]);
title("Portiloop data tested on Neural Network Trained ONLY on MASS (N=180)");
xlabel("Time (s)");
ylabel("a.u.");
fp = findobj('Color','r');
tn = findobj('Color','b');
tp = findobj('Color','g');
fn = findobj('Color','k');
v = [fp(1) tn(1) tp(1) fn(1)];
legend(v, ["False Positive", "True Negative", "True Positive", "False Negative"], 'location', 'best');

subplot(2, 1, 2)
hold on
seq_stride = 25;
plot_data = signal;
for tp=0:2000%length(state)-1
    idx = tp * seq_stride + 1;
    c = 'b';
    switch state_portiloop(tp+1)
        case 1
           c = 'g';
        case 2
            c = 'r';
        case 3
            c = 'k';
        case 4
            c = 'b';
    end
    plot(time_vect(idx:idx + seq_stride), plot_data(idx:idx + seq_stride), 'Color', c);
end
axis([120 150 -10 10]);
title("Portiloop data tested on Neural Network Trained ONLY on Portiloop data (N=1)");
xlabel("Time (s)");
ylabel("a.u.");
set(gcf,'color','w');