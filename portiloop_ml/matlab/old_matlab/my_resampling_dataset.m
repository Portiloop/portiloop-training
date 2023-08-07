%% load data
path = "../dataset/";
spindle_250 = load(path+"spindles_annotations_p1_big_at_250hz.txt");
data_256 = load(path+"dataset_p1_big_at_256_to_resample.txt");
%% resample
data_250_matlab = resample(data_256, 250, 256);
%% create vector
size_250 = size(spindle_250,1);
time_vect_250 = linspace(0,size_250/250, size_250);
%% generate vector
data_250_matlab = [data_250_matlab;0];
spindle_250 = [spindle_250;-3];
dataset = [data_250_matlab, spindle_250];
%% create file
writematrix(dataset, path+"dataset_p1_big_250_matlab.txt");