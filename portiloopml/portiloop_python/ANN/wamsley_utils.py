import copy
import numpy as np
from scipy.signal import fftconvolve


def get_spindle_onsets(indexes, sampling_rate=250, min_label_time=0.4):
    '''
    Given a list of positive indexes, return the start index of every spindle.

    :param positive_indexes: An array of size of signal with 1 at the index of the spindle and 0 elsewhere
    '''

    if len(indexes) < sampling_rate:
        return np.array([])

    # Calculate the minimum interval between two spindles
    interval = int(min_label_time * sampling_rate)

    indexes = np.array(indexes)
    # Create a shifted version of the array to check for the "0 followed by 1" condition
    shifted_indexes = np.roll(indexes, shift=1)

    # Find the indexes where 0 is followed by 1
    indexes = np.where((shifted_indexes == False) & (indexes == True))[0]

    # Remove the indexes that are too close together
    if indexes.size > 0:
        indexes = indexes[np.insert(np.diff(indexes) >= interval, 0, True)]

    return indexes


def binary_f1_score(baseline_index, model_index, sampling_rate=250, min_time_positive=0.4):
    tp = 0
    fp = 0
    fn = 0
    closest = []

    # Calculate the minimum interval between two spindles
    threshold = int(min_time_positive * sampling_rate)

    if len(baseline_index) == 0 or len(model_index) == 0:
        return 0, 0, 0, tp, fp, fn, closest

    for index in baseline_index:
        # How many in model are within a threshold distance of the baseline
        similarity = len(np.where(np.abs(model_index - index) < threshold)[0])
        closest.append(np.min(np.abs(model_index - index)))
        # If none, we have a false negative
        if similarity == 0:
            fn += 1
        # If one or more, we have a true positive
        else:
            tp += 1

    # To get the false positive, we take the number of indexes in model that are not in baseline
    fp = len(model_index) - tp

    assert tp + fp == len(model_index)
    assert tp + fn == len(baseline_index)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / ((precision + recall) + 1e-8)

    return precision, recall, f1, tp, fp, fn, closest


def detect_wamsley(data, mask, sampling_rate=250, thresholds=None, fixed=True):
    """
    Detect spindles in the data using the method described in Wamsley et al. 2012
    :param data: The data to detect spindles in
    :param mask: The mask to apply to the data to keep only N2, N3 sleep
    :param sampling_rate: The sampling rate of the data
    :param thresholds: The past thresholds to use for the moving average
    """
    if fixed:
        frequency = (11, 16)
    else:
        frequency = (12, 15)

    duration = (0.3, 3)
    wavelet_options = {'f0': np.mean(frequency),
                       'sd': .8,
                       'dur': 1.,
                       'output': 'complex'
                       }
    smooth_duration = .1
    det_thresh = 4.5
    merge_thresh = 0.3
    min_interval = 0.5  # minimum time in seconds between events

    thresholds = copy.deepcopy(thresholds)

    # First, we transform the signal using wavelet transform
    if mask is None:
        data_detect = data
    else:
        data_detect = data[mask]

    # If the data is too short, return an empty array
    if len(data_detect) <= 30 * 250:
        return np.array([]), None, None, None, thresholds

    if mask is None:
        timestamps = np.arange(0, len(data)) / sampling_rate
    else:
        timestamps = (np.arange(0, len(data)))[mask] / sampling_rate
    assert len(data_detect) == len(timestamps)
    data_detect = morlet_transform(data_detect, sampling_rate, wavelet_options)

    if fixed:
        data_detect = np.real(data_detect)
    else:
        data_detect = np.real(data_detect ** 2) ** 2

    # Then we smoothen out the signal
    data_detect = smooth(data_detect, sampling_rate, smooth_duration)

    if fixed:
        data_detect = np.abs(data_detect)

    mean_spindle_power = np.mean(data_detect)

    # Then, we define the threshold
    _threshold = det_thresh * mean_spindle_power
    if thresholds is None:
        threshold = _threshold
    else:
        thresholds.append((_threshold, len(data_detect)))

        # Compute the weighted average of the thresholds to get the current threshold
        weigths = np.array([i[1] for i in thresholds])
        weigths = weigths / np.sum(weigths)
        threshold = np.sum([i[0] * j for i, j in zip(thresholds, weigths)])

    # Then we find the peaks
    peaks = data_detect >= threshold

    # Get the start, end and peak power index of each spindle
    events = _detect_start_end(peaks)

    # If no events are found, return an empty array
    if events is None:
        return np.array([]), _threshold, threshold, data_detect, thresholds

    # add the location of the peak in the middle
    events = np.insert(events, 1, 0, axis=1)
    for i in events:
        i[1] = i[0] + np.argmax(data_detect[i[0]:i[2]])

    # Merge the events that are too close
    events = _merge_close(data_detect, events, timestamps, min_interval)
    # Filter the events based on duration
    events = within_duration(events, timestamps, duration)
    # Remove the events that straddle a stitch
    events = remove_straddlers(events, timestamps, sampling_rate)

    # Get the real indexes back
    def to_index(point):
        return timestamps[point] * sampling_rate

    if len(events) == 0:
        return np.array([]), _threshold, threshold, data_detect, thresholds

    events = np.vectorize(to_index)(events)

    return events.astype(int), _threshold, threshold, data_detect, thresholds


def merge_events(start_indexes, end_indexes, timestamps, threshold):
    merged_start_indexes = []
    merged_end_indexes = []

    # Sort the events based on timestamps
    events = list(zip(start_indexes, end_indexes))

    # Iterate over the sorted events and merge if close enough
    prev_start, prev_end = events[0]
    for start, end in events[1:]:
        if timestamps[start] - timestamps[prev_end] <= threshold:
            # Merge the events
            prev_end = end
        else:
            # Append the merged event
            merged_start_indexes.append(prev_start)
            merged_end_indexes.append(prev_end)
            # Update the previous event
            prev_start, prev_end = start, end

    # Append the last merged event
    merged_start_indexes.append(prev_start)
    merged_end_indexes.append(prev_end)

    return np.array(merged_start_indexes), np.array(merged_end_indexes)


def filter_events_time(start_indexes, end_indexes, timestamps, min_length, max_length):
    filtered_start_indexes = []
    filtered_end_indexes = []

    for start, end in zip(start_indexes, end_indexes):
        event_length = timestamps[end] - timestamps[start]

        if min_length <= event_length <= max_length:
            filtered_start_indexes.append(start)
            filtered_end_indexes.append(end)

    return np.array(filtered_start_indexes), np.array(filtered_end_indexes)


def _detect_start_end(true_values):
    """From ndarray of bool values, return intervals of True values.

    Parameters
    ----------
    true_values : ndarray (dtype='bool')
        array with bool values

    Returns
    -------
    ndarray (dtype='int')
        N x 2 matrix with starting and ending times.
    """
    neg = np.zeros((1), dtype='bool')
    int_values = np.asarray(np.concatenate((neg, true_values[:-1], neg)),
                            dtype='int')
    # must discard last value to avoid axis out of bounds
    cross_threshold = np.diff(int_values)

    event_starts = np.where(cross_threshold == 1)[0]
    event_ends = np.where(cross_threshold == -1)[0]

    if len(event_starts):
        events = np.vstack((event_starts, event_ends)).T

    else:
        events = None

    return events


def _merge_close(dat, events, time, min_interval):
    """Merge together events separated by less than a minimum interval.

    Parameters
    ----------
    dat : ndarray (dtype='float')
        vector with the data after selection-transformation
    events : ndarray (dtype='int')
        N x 3 matrix with start, peak, end samples
    time : ndarray (dtype='float')
        vector with time points
    min_interval : float
        minimum delay between consecutive events, in seconds

    Returns
    -------
    ndarray (dtype='int')
        N x 3 matrix with start, peak, end samples
    """
    if not events.any():
        return events

    no_merge = time[events[1:, 0] - 1] - time[events[:-1, 2]] >= min_interval

    if no_merge.any():
        begs = np.concatenate([[events[0, 0]], events[1:, 0][no_merge]])
        ends = np.concatenate([events[:-1, 2][no_merge], [events[-1, 2]]])

        new_events = np.vstack((begs, ends)).T
    else:
        new_events = np.asarray([[events[0, 0], events[-1, 2]]])

    # add the location of the peak in the middle
    new_events = np.insert(new_events, 1, 0, axis=1)
    for i in new_events:
        if i[2] - i[0] >= 1:
            i[1] = i[0] + np.argmax(dat[i[0]:i[2]])

    return new_events


def within_duration(events, time, limits):
    """Check whether event is within time limits.

    Parameters
    ----------
    events : ndarray (dtype='int')
        N x M matrix with start sample first and end samples last on M
    time : ndarray (dtype='float')
        vector with time points
    limits : tuple of float
        low and high limit for spindle duration

    Returns
    -------
    ndarray (dtype='int')
        N x M matrix with start sample first and end samples last on M
    """
    min_dur = max_dur = np.ones(events.shape[0], dtype=bool)

    if limits[0] is not None:
        min_dur = time[events[:, -1] - 1] - time[events[:, 0]] >= limits[0]

    if limits[1] is not None:
        max_dur = time[events[:, -1] - 1] - time[events[:, 0]] <= limits[1]

    return events[min_dur & max_dur, :]


def remove_straddlers(events, time, s_freq, tolerance=0.1):
    """Reject an event if it straddles a stitch, by comparing its 
    duration to its timespan.

    Parameters
    ----------
    events : ndarray (dtype='int')
        N x M matrix with start, ..., end samples
    time : ndarray (dtype='float')
        vector with time points
    s_freq : float
        sampling frequency
    tolerance : float, def=0.1
        maximum tolerated difference between event duration and timespan

    Returns
    -------
    ndarray (dtype='int')
        N x M matrix with start , ..., end samples
    """
    dur = (events[:, -1] - 1 - events[:, 0]) / s_freq
    continuous = time[events[:, -1] - 1] - time[events[:, 0]] - dur < tolerance

    return events[continuous, :]


def morlet_transform(data, s_freq, morlet_options):
    """ Adapted from Wonambi
    Computes the morlet transform of the data
    """
    f0 = morlet_options['f0']
    sd = morlet_options['sd']
    dur = morlet_options['dur']
    output = morlet_options['output']

    wm = _wmorlet(f0, sd, s_freq, dur)
    data = fftconvolve(data, wm, mode='same')
    if 'absolute' == output:
        data = np.absolute(data)

    return data


def _wmorlet(f0, sd, sampling_rate, ns=5):
    """Adapted from nitime

    Returns a complex morlet wavelet in the time domain

    Parameters
    ----------
        f0 : center frequency
        sd : standard deviation of frequency
        sampling_rate : samplingrate
        ns : window length in number of standard deviations

    Returns
    -------
    ndarray
        complex morlet wavelet in the time domain
    """
    st = 1. / (2. * np.pi * sd)
    w_sz = float(int(ns * st * sampling_rate))  # half time window size
    t = np.arange(-w_sz, w_sz + 1, dtype=float) / sampling_rate
    w = (np.exp(-t ** 2 / (2. * st ** 2)) * np.exp(2j * np.pi * f0 * t) /
         np.sqrt(np.sqrt(np.pi) * st * sampling_rate))
    return w


def smooth(data, dur, s_freq):
    """ Adapted from Wonambi
    Smoothen the data using a flat window
    """
    flat = np.ones(int(dur * s_freq))
    H = flat / sum(flat)
    data = fftconvolve(data, H, mode='same')
    return data


def RMS_score(candidate, Fs=250, lowcut=11, highcut=16):

    # Filter the signal
    stopbbanAtt = 60  # stopband attenuation of 60 dB.
    width = .5  # This sets the cutoff width in Hertz
    nyq = 0.5 * Fs
    ntaps, _ = kaiserord(stopbbanAtt, width/nyq)
    atten = kaiser_atten(ntaps, width/nyq)
    beta = kaiser_beta(atten)
    a = 1.0
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq,
                  pass_zero=False, window=('kaiser', beta), scale=False)
    filtered_signal = filtfilt(taps, a, candidate)

    # Get the baseline and the detection window for the RMS
    detect_index = len(candidate) // 2
    size_window = 0.5 * Fs
    baseline_idx = -2 * Fs  # Index compared to the detection window
    baseline = filtered_signal[detect_index +
                               baseline_idx:detect_index + baseline_idx + size_window]
    detection = filtered_signal[detect_index:detect_index + size_window]

    # Calculate the RMS
    baseline_rms = torch.sqrt(torch.mean(torch.square(baseline)))
    detection_rms = torch.sqrt(torch.mean(torch.square(detection)))

    score = detection_rms / baseline_rms
    return score
