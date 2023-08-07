path = "../dataset/";
phase = 'full';
size_data = 'big';
type = 'classification';
dataset = load(path+"dataset_classification_"+phase+"_"+size_data+"_250_matlab_standardized_envelope_pf.txt");
spindle = load(path+"spindles_annotations_"+type+"_"+ phase+"_"+size_data +"_at_250hz.txt");
datasetupdate = single([dataset(:,1), dataset(:,2), dataset(:,3), spindle]);

writematrix(datasetupdate, path+"dataset_"+type+"_"+phase + "_"+size_data+"_250_matlab_standardized_envelope_pf.txt");
