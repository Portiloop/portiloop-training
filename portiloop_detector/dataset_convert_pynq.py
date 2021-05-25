# convert test set into file to be read by the pynq
import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from portiloop_detector.experiments import path_experiment
from portiloop_detector_training import path_dataset, subject_list, LEN_SEGMENT, SignalDataset, filename_dataset

fe = 250


def convert_test_set():
    all_subject = pd.read_csv(Path(path_dataset) / subject_list, header=None, delim_whitespace=True).to_numpy()
    train_subject, test_subject = train_test_split(all_subject, train_size=0.9, random_state=0)
    train_subject, validation_subject = train_test_split(train_subject, train_size=0.95, random_state=0)  # with K fold cross validation, this
    # split will be done K times

    logging.debug(f"Subjects in training : {train_subject[:, 0]}")
    logging.debug(f"Subjects in validation : {validation_subject[:, 0]}")
    logging.debug(f"Subjects in test : {test_subject[:, 0]}")
    len_segment_s = LEN_SEGMENT * fe

    ds_test = SignalDataset(filename=filename_dataset,
                            path=path_dataset,
                            window_size=1,
                            fe=fe,
                            seq_len=1,
                            seq_stride=1,  # just to be sure, fixed value
                            list_subject=test_subject,
                            len_segment=len_segment_s)

    logging.debug(len(ds_test))
    pynq_dataset = ds_test.data[:,ds_test.indices]
    logging.debug(pynq_dataset.shape)
    logging.debug(len(pynq_dataset))
    with open(path_experiment/"testset.pkl", "wb") as file:
        pickle.dump(pynq_dataset, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--experiment_index', type=int)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    if args.output_file is not None:
        logging.basicConfig(format='%(levelname)s: %(message)s', filename=args.output_file, level=logging.DEBUG)
        logging.debug('This message should go to the log file')
        logging.info('So should this')
        logging.warning('And this, too')
        logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    convert_test_set()
