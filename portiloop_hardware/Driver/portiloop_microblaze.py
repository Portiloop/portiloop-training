import asyncio
import numpy as np
from pynq import Xlnk
from . import Arduino
from scipy.signal import lfilter
import time
import os
import csv

ARDUINO_PORTILOOP_MICROBLAZE = "portiloop_microblaze.bin" #file compiled with xilinx SDK, contain microblaze C code
SETUP_DEVICE = 0x1 #command to communicate with microblaze
READ_REGISTERS = 0x3
START_ACQUIRE = 0x5

NB_INPUT = 1 #should be fixed to 1 for now, until modification in microblaze
WINDOW_SIZE = 54
HIDDEN_VECTOR_SIZE = 7*1*NB_INPUT
TOT_INPUT_NN = NB_INPUT*WINDOW_SIZE + NB_INPUT*HIDDEN_VECTOR_SIZE
TOT_OUTPUT_NN = 1 + NB_INPUT*HIDDEN_VECTOR_SIZE

class Arduino_Portiloop_Microblaze(object):
    def __init__(self, ol, filename, nb_eeg = 1, nb_ecg = 0, nb_buffer = 2, filter_12_16 = 1, filter_lp_35 = 0, notch_filter_mode = 0):
        # notch_filter_mode : 0 = 60 Hz notch filter, 1 = 50 Hz notch filter
        self.microblaze = Arduino(ol.iop_arduino.mb_info, ARDUINO_PORTILOOP_MICROBLAZE)
        self.buf_manager = Xlnk()
        self.microblaze.write_blocking_command(SETUP_DEVICE) #start by setting up the ADS1299
        if(nb_eeg > 1): #should be change later to add new electrodes, modification must be made in xilinx SDK too
            nb_eeg = 1
        self.nb_eeg = nb_eeg
        self.nb_ecg = nb_ecg
        self.filename = filename #filename where every buffer will be written in
        self.nb_buffer = nb_buffer
        self.nb_lp_35 = filter_lp_35 #if 1 : filter the first electrode only
        self.nb_12_16 = filter_12_16 #if 2 : filter the first two electrodes
        if self.nb_eeg<self.nb_12_16:
            self.nb_12_16 = self.nb_eeg
        if self.nb_eeg<self.nb_lp_35:
            self.nb_lp_35 = self.nb_eeg
        if not os.path.exists("../Data/" + filename):
            os.makedirs("../Data/" + filename)
        self.audio= ol.my_audio.adau1761_0 #configure audio device
        self.audio.select_microphone()
        self.audio.load("sine.wav") #select the audio file to play
        self.notch_filter_mode = notch_filter_mode


    def read_register(self):
        self.microblaze.write_blocking_command(READ_REGISTERS)
        return self.microblaze.read_mailbox(0,26)
    
    @asyncio.coroutine
    def interrupt_handler_async(self):
        if self.microblaze.interrupt is None:
            raise RuntimeError('Interrupts not available in this Overlay')
        cycle = 0
        packet = 0
        val = 0
        tot_time = 0
        tmax = 0

        while(1):
            yield from self.microblaze.interrupt.wait() # Wait for interrupt
            self.microblaze.interrupt.clear()
            start_time = time.time()
            data_ele = []
            timestamp = []
            filtered_12_16 = []
            filtered_lp_35 = []
            #read data from the shared buffer
            for j in range(self.nb_eeg+ self.nb_ecg):
                data_ele.append(self.data_buffer[val][j][0])
                data_ele[j] = data_ele[j]*(4.5/24)/(2**23-1)
                nb_acq = 1
                if j < self.nb_12_16:
                    filtered_12_16.append(self.data_buffer[val][j][nb_acq])
                    nb_acq += 1
                if j < self.nb_lp_35:
                    filtered_lp_35.append(self.data_buffer[val][j][nb_acq])
                    nb_acq += 1
                if j < self.nb_eeg:
                    timestamp.append(self.data_buffer[val][j][nb_acq])
                    nb_acq += 1
            #put them in files
            for i in range(self.length):
                for k in range(self.nb_eeg):
                    self.data_file[k].write(str(data_ele[k][i]) + "\n")
                    if i < (self.length/2):
                        if k < self.nb_12_16:
                            self.filtered_12_16_file[k].write(str(filtered_12_16[k][i]) + "\n")
                        if k < self.nb_lp_35:
                            self.filtered_lp_35_file[k].write(str(filtered_lp_35[k][i]) + "\n")
                        self.timestamp_file[k].write(str(timestamp[k][i]) + "\n")
                for k in range(self.nb_ecg):
                    self.ecg_file[k].write(str(data_ele[self.nb_eeg+k][i]) + "\n")
            val = (val + 1)%self.nb_buffer
            packet = packet + 1
            stop_time = time.time()
            t = stop_time - start_time
            tot_time = tot_time + t
            if tmax < t:
                tmax = t
            #when the file is full, close in and use the next one
            if packet == self.nb_packet:
                for i in range(self.nb_eeg):
                    self.data_file[i].close()
                    self.timestamp_file[i].close()
                    if i < self.nb_12_16:
                        self.filtered_12_16_file[i].close()
                    if i < self.nb_lp_35:
                        self.filtered_lp_35_file[i].close()
                for i in range(self.nb_ecg):
                    self.ecg_file[i].close()
                print("Episode ", cycle)
                print("t mean " + str(tot_time/packet))
                tot_time = 0
                print("tmax " + str(tmax))
                tmax = 0
                cycle = cycle+1
                packet = 0
                if(cycle == self.nb_cycles):#when acquisition is over
                    break;
                for i in range(self.nb_eeg):
                    self.data_file[i] = open("../Data/"+self.filename+"/eeg_"+ str(i) + "_data_" + str(cycle) + ".txt", "w", encoding="utf-8")
                    self.timestamp_file[i] = open("../Data/"+self.filename+"/eeg_"+ str(i) + "_timestamp_" + str(cycle) + ".txt", "w", encoding="utf-8")
                    if i < self.nb_12_16:
                        self.filtered_12_16_file[i] = open("../Data/"+self.filename+"/eeg_" + str(i) + "_filtered_12_16_"+ str(cycle) + ".txt", "w", encoding="utf-8")
                    if i < self.nb_lp_35:
                        self.filtered_lp_35_file[i] = open("../Data/"+self.filename+"/eeg_" + str(i) + "_filtered_lp_35_"+ str(cycle) + ".txt", "w", encoding="utf-8")
                for i in range(self.nb_ecg):
                    self.ecg_file[i] = open("../Data/"+self.filename+"/ecg_"+ str(i) + "_" + str(cycle) + ".txt", "w", encoding="utf-8")
        self.shared_buff1.freebuffer()
        self.shared_buff2.freebuffer()
        self.audio.playend()
        self.audio_buff.freebuffer()

        for j in range(self.nb_buffer):
            for i in range(self.nb_eeg + self.nb_ecg):
                self.data_buffer[j][i][0].freebuffer()
                nb_acq = 1
                if i < self.nb_12_16:
                    self.data_buffer[j][i][nb_acq].freebuffer()
                    nb_acq += 1
                if i < self.nb_lp_35:
                    self.data_buffer[j][i][nb_acq].freebuffer()
                    nb_acq += 1
                if i < self.nb_eeg:
                    self.data_buffer[j][i][nb_acq].freebuffer()
                    nb_acq += 1
    def acquire_data(self,length_buff_ms = 250, length_buffer_minutes = 30, nb_cycles = 4):
        self.nb_cycles = nb_cycles
        self.length = int(length_buff_ms/2)
        self.nb_packet = int(length_buffer_minutes*500*60/self.length)
        self.data_file = []
        self.timestamp_file = []
        self.filtered_12_16_file = []
        self.filtered_lp_35_file = []
        self.ecg_file = []
        #create the files
        for i in range(self.nb_eeg):
            self.data_file.append(open("../Data/"+self.filename+"/eeg_"+ str(i) + "_data_0.txt", "w", encoding="utf-8"))
            self.timestamp_file.append(open("../Data/"+self.filename+"/eeg_"+ str(i) + "_timestamp_0.txt", "w", encoding="utf-8"))
            if i < self.nb_12_16:
                self.filtered_12_16_file.append(open("../Data/"+self.filename+"/eeg_" + str(i) + "_filtered_12_16_0.txt", "w", encoding="utf-8"))
            if i < self.nb_lp_35:
                self.filtered_lp_35_file.append(open("../Data/"+self.filename+"/eeg_" + str(i) + "_filtered_lp_35_0.txt", "w", encoding="utf-8"))
        for i in range(self.nb_ecg):
            self.ecg_file.append(open("../Data/"+self.filename+"/ecg_"+ str(i) + "_0.txt", "w", encoding="utf-8"))
        #create every buffer used and shared with microblaze
        self.shared_buff1 = self.buf_manager.cma_array(1, dtype = np.int32)
        self.shared_buff2 = self.buf_manager.cma_array(1, dtype = np.int32)
        self.audio_buff = self.buf_manager.cma_array(self.audio.buffer.shape, dtype = np.int32)
        self.audio_buff[:] = self.audio.buffer
        self.nn_input_buff = self.buf_manager.cma_array(TOT_INPUT_NN, dtype = np.float32)
        self.nn_output_buff = self.buf_manager.cma_array(TOT_OUTPUT_NN, dtype = np.float32)
        data = [self.nb_ecg, self.nb_eeg, self.length, int(self.nb_cycles*self.nb_packet), self.notch_filter_mode, self.nb_buffer, self.shared_buff1.physical_address, self.shared_buff2.physical_address, self.nb_12_16, self.nb_lp_35,self.audio_buff.physical_address, self.audio.buffer.shape[0], self.nn_input_buff.physical_address, self.nn_output_buff.physical_address]

        self.data_buffer = []
        for j in range(self.nb_buffer):
            self.data_buffer.append([])
            for i in range(self.nb_eeg + self.nb_ecg):
                self.data_buffer[j].append([])
                self.data_buffer[j][i].append(self.buf_manager.cma_array(self.length, dtype = np.float32))
                data.append(self.data_buffer[j][i][0].physical_address)
                nb_acq = 1
                if i < self.nb_12_16:
                    self.data_buffer[j][i].append(self.buf_manager.cma_array(int(self.length/2), dtype = np.float32))
                    data.append(self.data_buffer[j][i][nb_acq].physical_address)
                    nb_acq += 1
                if i < self.nb_lp_35:
                    self.data_buffer[j][i].append(self.buf_manager.cma_array(int(self.length/2), dtype = np.float32))
                    data.append(self.data_buffer[j][i][nb_acq].physical_address)
                    nb_acq += 1
                if i < self.nb_eeg:
                    self.data_buffer[j][i].append(self.buf_manager.cma_array(int(self.length/2), dtype = np.float32))#timestamp buff
                    data.append(self.data_buffer[j][i][nb_acq].physical_address)
                    nb_acq += 1

        self.audio.playinit()
        self.microblaze.interrupt.clear()
        self.microblaze.write_mailbox(0,data) #send the data to microblaze
        self.microblaze.write_non_blocking_command(START_ACQUIRE)
        loop = asyncio.get_event_loop()#launch the interupt function
        loop.run_until_complete(asyncio.ensure_future(
            self.interrupt_handler_async()
        ))
