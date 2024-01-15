import gradio as gr
import plotly.express as px
import numpy as np
import pandas as pd
import pyxdf
from datetime import datetime


STREAM_NAMES = {
    'filtered_data': 'Portiloop Filtered',
    'raw_data': 'Portiloop Raw Data',
    'stimuli': 'Portiloop_stimuli'
}


def xdf2df(xdf_path):
    '''
    Load an XDF file and return a pandas dataframe
    '''
    xdf_data, info = pyxdf.load_xdf(xdf_path)

    # Get the datetime of the recording in timestamp
    rec_date = datetime.strptime(
        info['info']['datetime'][0], "%Y-%m-%dT%H:%M:%S-%f")

    # Load all streams given their names
    filtered_stream, raw_stream, markers = None, None, []
    for stream in xdf_data:
        # print(stream['info']['name'])
        if stream['info']['name'][0] == STREAM_NAMES['filtered_data']:
            filtered_stream = stream
        elif stream['info']['name'][0] == STREAM_NAMES['raw_data']:
            raw_stream = stream
        elif STREAM_NAMES['stimuli'] in stream['info']['name'][0]:
            markers.append(stream)

    if filtered_stream is None or raw_stream is None:
        raise ValueError(
            "One of the necessary streams could not be found. Make sure that at least one signal stream is present in XDF recording")

    timestamps = filtered_stream['time_stamps']
    # Convert all the timestamps to datetime objects
    timestamps_datetime = [rec_date +
                           pd.Timedelta(seconds=ts) for ts in timestamps]

    filtered_data = filtered_stream['time_series']
    raw_data = raw_stream['time_series']

    assert len(timestamps) == len(filtered_data) == len(raw_data)

    # Create a dataframe with the data
    df = pd.DataFrame({'time': timestamps_datetime})

    # Add the filtered data
    for i in range(filtered_data.shape[1]):
        df[f'filtered_{i}'] = filtered_data[:, i]

    # Add the raw data
    for i in range(raw_data.shape[1]):
        df[f'raw_{i}'] = raw_data[:, i]

    # For each marker stream, get the timestamps and the marker values
    for marker in markers:
        marker_timestamps = marker['time_stamps']
        # Convert all the timestamps to datetime objects
        marker_values = marker['time_series']
        marker_name = marker['info']['name'][0]

        # Create a list of empty strings
        marker_list = [''] * len(timestamps)

        # For each marker value, find the closest timestamp and add the marker
        # value to the list
        # for i in range(len(marker_timestamps)):
        #     idx = np.argmin(np.abs(timestamps - marker_timestamps[i]))
        #     marker_list[idx] = marker_values[i][0]

        df[marker_name] = marker_list

    return df


class EEGFile:
    def __init__(self):
        self.time_display = 0
        self.increment = 60 * 250 * 5

    def load(self, file_path):
        if file_path.name.split(".")[-1] not in ['xdf', 'csv', 'edf']:
            raise gr.Error("Please upload a file with a supported format.")
        else:
            self.file_path = file_path.name

        # print(f"Loading file {self.file_path}...")

        # # Load the data:
        # self.data = xdf2df(self.file_path)
        # self.visualize_eeg()
        # print("Done...")

        return self.file_path # , self.fig

    def visualize_eeg(self):
        # Select the desired column which has the data
        timestamps_column = 'time'
        voltage_column = 'filtered_0'
        stimulus_column = 'Portiloop_stimuli'

        # Now plot your data
        fig = px.line(self.data[self.time_display:self.time_display + self.increment], x=timestamps_column, y=voltage_column,
                      title='EEG Data Visualization', render_mode='webg1')

        # Add vertical line for spindles at the right timestamps
        spindle_idx = np.where(self.data[stimulus_column] != '')[0]
        for idx in spindle_idx:
            fig.add_vline(x=self.data[timestamps_column][idx], line_width=1, line_dash="dash",
                          line_color="red")

        fig.update_xaxes(title='Time')
        fig.update_yaxes(title='Voltage')

        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1s",
                             step="second",
                             stepmode="backward"),
                        dict(count=5,
                             label="5s",
                             step="second",
                             stepmode="backward"),
                        dict(count=10,
                             label="10s",
                             step="second",
                             stepmode="backward"),
                        dict(count=30,
                             label="30s",
                             step="second",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            ),
            dragmode="pan",
        )

        self.fig = fig

    def increment_time_display(self, button):
        self.time_display += self.increment
        self.visualize_eeg()
        return self.fig

    def decrement_time_display(self, button):
        self.time_display -= self.increment
        self.visualize_eeg()
        return self.fig


eeg_test = EEGFile()

with gr.Blocks() as demo:

    eeg_file = gr.File(label="EEG File (xdf, edf, csv)")

    btn = gr.Button(value="Run Model")
    outputs = gr.Plot()
    with gr.Row():
        prev_button = gr.Button(value="Previous")
        next_button = gr.Button(value="Next")
        
    prev_button.click(eeg_test.decrement_time_display, prev_button, outputs)
    next_button.click(eeg_test.increment_time_display, next_button, outputs)

if __name__ == "__main__":
    print("testing heere")
    demo.launch(server_port=8000)
    print("done launching")
