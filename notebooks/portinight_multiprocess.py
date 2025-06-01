import os
import pandas as pd
import numpy as np
from multiprocessing import Process, Manager
from scipy.signal import firwin
from tqdm import tqdm
from portiloopml.portiloop_python.ANN.wamsley_utils import detect_lacourse


def shift_numpy(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


class FIR:
    def __init__(self, nb_channels, coefficients, buffer=None):

        self.coefficients = np.expand_dims(np.array(coefficients), axis=1)
        self.taps = len(self.coefficients)
        self.nb_channels = nb_channels
        self.buffer = np.array(buffer) if buffer is not None else np.zeros(
            (self.taps, self.nb_channels))

    def filter(self, x):
        self.buffer = shift_numpy(self.buffer, 1, x)
        filtered = np.sum(self.buffer * self.coefficients, axis=0)
        return filtered


class FilterPipeline:
    def __init__(self,
                 nb_channels,
                 sampling_rate,
                 power_line_fq=60,
                 use_custom_fir=False,
                 custom_fir_order=20,
                 custom_fir_cutoff=30,
                 alpha_avg=0.1,
                 alpha_std=0.001,
                 epsilon=0.000001,
                 filter_args=[]):
        if len(filter_args) > 0:
            use_fir, use_notch, use_std = filter_args
        else:
            use_fir = True,
            use_notch = False,
            use_std = True
        self.use_fir = use_fir
        self.use_notch = use_notch
        self.use_std = use_std
        self.nb_channels = nb_channels
        assert power_line_fq in [
            50, 60], f"The only supported power line frequencies are 50 Hz and 60 Hz"
        if power_line_fq == 60:
            self.notch_coeff1 = -0.12478308884588535
            self.notch_coeff2 = 0.98729186796473023
            self.notch_coeff3 = 0.99364593398236511
            self.notch_coeff4 = -0.12478308884588535
            self.notch_coeff5 = 0.99364593398236511
        else:
            self.notch_coeff1 = -0.61410695998423581
            self.notch_coeff2 = 0.98729186796473023
            self.notch_coeff3 = 0.99364593398236511
            self.notch_coeff4 = -0.61410695998423581
            self.notch_coeff5 = 0.99364593398236511
        self.dfs = [np.zeros(self.nb_channels), np.zeros(self.nb_channels)]

        self.moving_average = None
        self.moving_variance = np.zeros(self.nb_channels)
        self.ALPHA_AVG = alpha_avg
        self.ALPHA_STD = alpha_std
        self.EPSILON = epsilon

        if use_custom_fir:
            self.fir_coef = firwin(
                numtaps=custom_fir_order+1, cutoff=custom_fir_cutoff, fs=sampling_rate)
        else:
            self.fir_coef = [
                0.001623780150148094927192721215192250384,
                0.014988684599373741992978104065059596905,
                0.021287595318265635502275046064823982306,
                0.007349500393709578957568417933998716762,
                -0.025127515717112181709014251396183681209,
                -0.052210507359822452833064687638398027048,
                -0.039273839505489904766477593511808663607,
                0.033021568427940004020193498490698402748,
                0.147606943281569008563636202779889572412,
                0.254000252034505602516389899392379447818,
                0.297330876398883392486283128164359368384,
                0.254000252034505602516389899392379447818,
                0.147606943281569008563636202779889572412,
                0.033021568427940004020193498490698402748,
                -0.039273839505489904766477593511808663607,
                -0.052210507359822452833064687638398027048,
                -0.025127515717112181709014251396183681209,
                0.007349500393709578957568417933998716762,
                0.021287595318265635502275046064823982306,
                0.014988684599373741992978104065059596905,
                0.001623780150148094927192721215192250384]
        self.fir = FIR(self.nb_channels, self.fir_coef)

    def filter(self, value):
        """
        value: a numpy array of shape (data series, channels)
        """
        for i, x in enumerate(value):  # loop over the data series
            # FIR:
            if self.use_fir:
                x = self.fir.filter(x)
            # notch:
            if self.use_notch:
                denAccum = (x - self.notch_coeff1 *
                            self.dfs[0]) - self.notch_coeff2 * self.dfs[1]
                x = (self.notch_coeff3 * denAccum + self.notch_coeff4 *
                     self.dfs[0]) + self.notch_coeff5 * self.dfs[1]
                self.dfs[1] = self.dfs[0]
                self.dfs[0] = denAccum
            # standardization:
            if self.use_std:
                if self.moving_average is not None:
                    delta = x - self.moving_average
                    self.moving_average = self.moving_average + self.ALPHA_AVG * delta
                    self.moving_variance = (
                        1 - self.ALPHA_STD) * (self.moving_variance + self.ALPHA_STD * delta**2)
                    moving_std = np.sqrt(self.moving_variance)
                    x = (x - self.moving_average) / (moving_std + self.EPSILON)
                else:
                    self.moving_average = x
            try:
                value[i] = x
            except:
                print(f"Error in filtering: {x}")
                continue
        return value


def online_detrend(y, alpha=0.95):
    detrended_y = np.zeros_like(y)
    trend = 0
    for i in range(len(y)):
        trend = alpha * trend + (1 - alpha) * y[i]
        detrended_y[i] = y[i] - trend
    return detrended_y


def raw2filtered(raw):
    '''
    Take in the raw data and filter it online, detrend it
    '''
    filtering4lac = FilterPipeline(
        nb_channels=1, sampling_rate=250, filter_args=[True, True, False])
    filtered4lac = []
    print(f'Filtering Data Online')
    for i in tqdm(raw):
        filtered4lac.append(filtering4lac.filter(np.array([i])))

    print(f'Detrending Data')
    detrended_data = online_detrend(np.array(filtered4lac).flatten())

    # Remove all values where tha absolute value is above 500
    detrended_data[np.abs(detrended_data) > 500] = 0

    # count the number of NaNs in the detrended data
    nans = np.sum(np.isnan(detrended_data))
    print(f'Number of NaNs in the detrended data: {nans}')

    # print(f"Running Lacourse")
    # data_detect = np.array(detrended_data)
    # mask = np.ones(len(data_detect), dtype=bool)
    # lacourse = detect_lacourse(
    #     data_detect,
    #     mask,
    #     sampling_rate=250,
    # )

    # nans = np.sum(np.isnan(detrended_data))
    # print(f'######Number of NaNs in the detrended data After Lacourse: {nans}')

    # print(f"########Lacourse found {len(lacourse)} spindles")
    lacourse = []

    # if len(lacourse) == 0:
    #     return None, None, None

    # print(f"Lacourse found {len(lacourse)} spindles")

    # Filter data online like on Portiloop
    # print(f"Filtering Online with standardization")
    # filtering_online = FilterPipeline(nb_channels=1, sampling_rate=250, filter_args=[True, True, True])
    # filtered_online = []
    # for i in tqdm(raw):
    #     filtered_online.append(filtering_online.filter(np.array([i])))
    # filtered_online = np.array(detrended_data).flatten()

    return detrended_data, lacourse, detrended_data


def process_file(file, path, age, gender, data_dict):
    filename = file.split('_')[:-1]
    filename = '_'.join(filename)

    df = pd.read_csv(os.path.join(path, file),
                     on_bad_lines='warn', encoding_errors='ignore')
    df['converted'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df['converted'] = df['converted'].fillna(method='ffill')
    useful_data = df['converted'].values
    filtered4lac, lacourse, filtered_online = raw2filtered(useful_data)

    spindle_info_mass = {}
    spindle_info_mass[filename] = {
        'onsets': [],
        'offsets': [],
        'labels_num': []
    }

    for spindle in lacourse:
        spindle_info_mass[filename]['onsets'].append(spindle[0])
        spindle_info_mass[filename]['offsets'].append(spindle[1])
        spindle_info_mass[filename]['labels_num'].append(1)

    data_dict[filename] = {
        'age': age,
        'gender': gender,
        'signal_mass': filtered4lac,
        'signal_filt': filtered_online,
        'ss_label': np.ones(len(filtered4lac)) * 5,
        'spindle_mass_lacourse': spindle_info_mass
    }


def do_one_subject(path, subject_id):
    save_path = '/project/portinight-dataset/'
    # subject_id = 'PN_07_CB'
    age = 25
    gender = 'F'

    # Create a dictionary to hold the processed data
    manager = Manager()
    data_dict = manager.dict()

    # Iterate through all the csvs in the folder
    processes = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            print(f"Processing {file}")
            p = Process(target=process_file, args=(
                file, path, age, gender, data_dict))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    # Save the collected data
    print(f"Saving {subject_id}.npz")
    np.savez_compressed(os.path.join(
        save_path, f"{subject_id}.npz"), dict(data_dict))

    print(f"All files processed.")

if __name__ == "__main__":
    path = '/project/portinight-raw'

    subjects = os.listdir(path)

    print(f"Processing all subjects: {subjects}")

    # for each subject in path
    for subject in subjects:
        if os.path.isdir(os.path.join(path, subject)):
            print(f"|||||||||||   Processing {subject} ||||||||||||")
            do_one_subject(os.path.join(path, subject), subject)
            print(f"|||||||||||   Finished {subject} ||||||||||||")
    print(f"Finished all subjects")