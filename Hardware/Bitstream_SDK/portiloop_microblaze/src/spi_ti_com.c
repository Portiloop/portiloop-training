#include <xparameters.h>
#include "spi.h"
#include "spi_ti_com.h"

//this code comes from spi.c : everything is copypaste except some modifications in mytransfert and mytransfertfaster
#ifdef XPAR_XSPI_NUM_INSTANCES
#include "xspi_l.h"
#include "xspi.h"


static XSpi xspi[XPAR_XSPI_NUM_INSTANCES];
XSpi *xspi_ptr = &xspi[0];
extern XSpi_Config XSpi_ConfigTable[];

spi spi_open_device(unsigned int device) {
	int status;
	u16 dev_id;
	unsigned int base_address;
	u32 control;

	if (device < XPAR_XSPI_NUM_INSTANCES) {
		dev_id = (u16) device;
	}
	else {
		int found = 0;
		for (u16 i = 0; i < XPAR_XSPI_NUM_INSTANCES; ++i) {
			if (XSpi_ConfigTable[i].BaseAddress == device) {
				found = 1;
				dev_id = i;
				break;
			}
		}
		if (!found)
			return -1;
	}
	status = XSpi_Initialize(&xspi[dev_id], dev_id);
	if (status != XST_SUCCESS) {
		return -1;
	}
	base_address = xspi[dev_id].BaseAddr;
	// Soft reset SPI
	XSpi_WriteReg(base_address, XSP_SRR_OFFSET, 0xA);
	// Master mode
	control = XSpi_ReadReg(base_address, XSP_CR_OFFSET);
	// Master Mode
	control |= XSP_CR_MASTER_MODE_MASK;
	// Enable SPI
	control |= XSP_CR_ENABLE_MASK;
	// Slave select manually
	control |= XSP_INTR_SLAVE_MODE_MASK;
	// Enable Transmitter
	control &= ~XSP_CR_TRANS_INHIBIT_MASK;
	// Write configuration word
	XSpi_WriteReg(base_address, XSP_CR_OFFSET, control);

	return (spi) dev_id;
}


#ifdef XPAR_IO_SWITCH_NUM_INSTANCES
#ifdef XPAR_IO_SWITCH_0_SPI0_BASEADDR
#include "xio_switch.h"
static int last_spiclk = -1;
static int last_miso = -1;
static int last_mosi = -1;
static int last_ss = -1;

spi spi_open(unsigned int spiclk, unsigned int miso,
		unsigned int mosi, unsigned int ss) {
	if (last_spiclk != -1)
		set_pin(last_spiclk, GPIO);
	if (last_miso != -1)
		set_pin(last_miso, GPIO);
	if (last_mosi != -1)
		set_pin(last_mosi, GPIO);
	if (last_ss != -1)
		set_pin(last_ss, GPIO);
	last_spiclk = spiclk;
	last_miso = miso;
	last_mosi = mosi;
	last_ss = ss;
	set_pin(spiclk, SPICLK0);
	set_pin(miso, MISO0);
	set_pin(mosi, MOSI0);
	set_pin(ss, SS0);
	return spi_open_device(XPAR_IO_SWITCH_0_SPI0_BASEADDR);
}
#endif
#endif


spi spi_configure(spi dev_id, unsigned int clk_phase,
		unsigned int clk_polarity) {
	u32 control;
	unsigned int base_address = xspi[dev_id].BaseAddr;
	// Soft reset SPI
	XSpi_WriteReg(base_address, XSP_SRR_OFFSET, 0xA);
	// Master mode
	control = XSpi_ReadReg(base_address, XSP_CR_OFFSET);
	// Master Mode
	control |= XSP_CR_MASTER_MODE_MASK;
	// Enable SPI
	control |= XSP_CR_ENABLE_MASK;
	// Slave select manually
	control |= XSP_INTR_SLAVE_MODE_MASK;
	// Enable Transmitter
	control &= ~XSP_CR_TRANS_INHIBIT_MASK;
	// XSP_CR_CLK_PHASE_MASK
	if (clk_phase) {
		control |= XSP_CR_CLK_PHASE_MASK;
	}
	// XSP_CR_CLK_POLARITY_MASK
	if (clk_polarity) {
		control |= XSP_CR_CLK_POLARITY_MASK;
	}
	// Write configuration word
	XSpi_WriteReg(base_address, XSP_CR_OFFSET, control);
	return dev_id;
}

void spi_close(spi dev_id) {
	XSpi_ClearStats(&xspi[dev_id]);
}


unsigned int spi_get_num_devices(void) {
	return XPAR_XSPI_NUM_INSTANCES;
}
//end of copy past part
//my part :
void my_spi_transfer(spi dev_id, const unsigned char *write_data, unsigned char *read_data,
		unsigned int length) {
	unsigned int i;
	unsigned volatile int j;
	unsigned int base_address = xspi[dev_id].BaseAddr;

	XSpi_WriteReg(base_address, XSP_SSR_OFFSET, 0xfe);
	for (i = 0; i < length; i++) {
		XSpi_WriteReg(base_address, XSP_DTR_OFFSET, write_data[i]);
		j = 400;//modification from the original code
		while (j--);
	}
	while (((XSpi_ReadReg(base_address, XSP_SR_OFFSET) & 0x04)) != 0x04);
	// delay for 10 clock cycles
	j = 10;
	while (j--);
	for (i = 0; i < length; i++) {
		read_data[i] = XSpi_ReadReg(base_address, XSP_DRR_OFFSET);
	}
	XSpi_WriteReg(base_address, XSP_SSR_OFFSET, 0xff);
}
void my_spi_transfer_faster(spi dev_id, const unsigned char *write_data, unsigned char *read_data,
		unsigned int length) {
	unsigned int i;
	unsigned volatile int j;
	unsigned int base_address = xspi[dev_id].BaseAddr;

	XSpi_WriteReg(base_address, XSP_SSR_OFFSET, 0xfe);
	for (i = 0; i < length; i++) {
		XSpi_WriteReg(base_address, XSP_DTR_OFFSET, write_data[i]);
	}
	while (((XSpi_ReadReg(base_address, XSP_SR_OFFSET) & 0x04)) != 0x04);
	// delay for 10 clock cycles
	j = 10;
	while (j--);
	for (i = 0; i < length; i++) {
		read_data[i] = XSpi_ReadReg(base_address, XSP_DRR_OFFSET);
	}
	XSpi_WriteReg(base_address, XSP_SSR_OFFSET, 0xff);
}

#endif
spi setup_spi() {
	spi device = spi_open(13, 12, 11, 10);
	device = spi_configure(device, 1, 0);
	return device;
}
spi configure_reg(spi device)
{
	unsigned char buf_write[16];//timing à respecter?
	unsigned char buf_read[16];
	buf_write[0] = SDATAC;
	buf_write[1] = WREG;
	buf_write[2] = 0x0B;
	buf_write[3] = 0x3e;
	buf_write[4] = 0x95;//0x96 : 250 Hz, 0x95 : 500 Hz
	buf_write[5] = 0xc0;
	buf_write[6] = 0xe1;
	buf_write[7] = 0x0;
	buf_write[8] = 0x60;
	buf_write[9] = 0x60;
	buf_write[10] = 0x60;
	buf_write[11] = 0x60;
	buf_write[12] = 0x60;
	buf_write[13] = 0x60;
	buf_write[14] = 0x60;
	buf_write[15] = RDATAC;
	my_spi_transfer(device, buf_write, buf_read, 16);//impossible to write all the register in one time, so split the task in two
	buf_write[0] = SDATAC;
	buf_write[1] = WREG+0x0C;
	buf_write[2] = 0x09;
	buf_write[3] = 0xe0;
	buf_write[4] = 0x0;
	buf_write[5] = 0x0;
	buf_write[6] = 0xff;
	buf_write[7] = 0xff;
	buf_write[8] = 0x0;
	buf_write[9] = 0x0;
	buf_write[10] = 0x01;
	buf_write[11] = 0x0;
	buf_write[12] = 0x20;
	buf_write[13] = 0x0;
	buf_write[14] = RDATAC;
	my_spi_transfer(device, buf_write, buf_read, 15);
	return device;
}

spi setup_my_device()
{
	spi device = setup_spi();
	device = configure_reg(device);
	return device;
}

void read_register(spi device)
{
	unsigned char buf_write[16];
	unsigned char buf_read[16];
	buf_write[0] = SDATAC;
	buf_write[1] = RREG;
	buf_write[2] = 0x0B;
	buf_write[3] = 0;
	buf_write[4] = 0;
	buf_write[5] = 0;
	buf_write[6] = 0;
	buf_write[7] = 0;
	buf_write[8] = 0;
	buf_write[9] = 0;
	buf_write[10] = 0;
	buf_write[11] = 0;
	buf_write[12] = 0;
	buf_write[13] = 0;
	buf_write[14] = 0;
	buf_write[15] = RDATAC;
	my_spi_transfer(device, buf_write, buf_read, 16);
	for(int j = 3; j < 15; ++j)
	{
		MAILBOX_DATA(j-3) = buf_read[j];
	}

	buf_write[0] = SDATAC;
	buf_write[1] = RREG+0x0C;
	buf_write[2] = 0x0a;
	buf_write[3] = 0;
	buf_write[4] = 0;
	buf_write[5] = 0;
	buf_write[6] = 0;
	buf_write[7] = 0;
	buf_write[8] = 0;
	buf_write[9] = 0;
	buf_write[10] = 0;
	buf_write[11] = 0;
	buf_write[12] = 0;
	buf_write[13] = 0;
	buf_write[14] = 0;
	buf_write[15] = RDATAC;
	my_spi_transfer(device, buf_write, buf_read, 16);

	for(int j = 3; j < 15; ++j)
	{
		MAILBOX_DATA(j-3+12) = buf_read[j]; // send the register read to the jupyter notebook
	}
}

void swap_buffer(int id)
{
	active_buff = buffList[id];
	pos = 0;
}


void start_acquire(int l, int nb_cycles, spi device, int nb_buffer, int nb_ecg,int nb_eeg,int nb_12_16,int nb_lp_30)
{
	int j;
	int value = 0;
	int res;
	pos = 0;
	int cycle = 0;
	int nb_acq;
	float timestamp;
	int last_audio = 0;
	unsigned int total_pos = 0;
	float moving_average_lp35 = 0; //moving average for the low-pass filtered signal at 30 Hz
	float moving_variance_lp35 = 0;
	float moving_std_lp35 = 0;
	float res_lp35;
	float delta_lp35;
	float moving_average_bp1216 = 0; //moving average for the bandpass signal filtered between 12 and 16 Hz
	float moving_variance_bp1216 = 0; //moving variance for the bandpass signal
	float moving_std_bp1216 = 0; //moving standard deviation for the bandpass signal
	float res_bp1216;
	float delta_bp1216;
	float moving_average_envelope = 0;
	float res_envelope;
	float delta_envelope;
	float DigitalFilter_states[2] = {};

	float moving_window_input1[WINDOW_SIZE] = {};
	float moving_window_input2[WINDOW_SIZE] = {};
	float hidden_vector[NB_PARALLELE][HIDDEN_VECTOR_SIZE] = {};

	for(int i = 0; i < HIDDEN_VECTOR_SIZE; ++i)
	{
		*(nn_input_buff+i+2*WINDOW_SIZE) = 0; //hidden size initialization at 0
	}

	unsigned int moving_idx = 0;
	unsigned int nn_res_received = 1;
	unsigned int idx_sample = 0;
	unsigned int pos_sample[NB_PARALLELE] = {};

	unsigned int wait_stim = 0;
	unsigned int in_spindle = 0;

	unsigned char buf_read[4];
	unsigned char buf_write[1];
	buf_write[0] = START;
	my_spi_transfer(device, buf_write, buf_read, 1);
	gpio DRDY = gpio_open(9);
	gpio_set_direction(DRDY, GPIO_IN);

	if (TRIGGER_STIMULATION != 0) //if we want to use trigger instead of sending audio, we setup a GPIO
	{
		gpio trigger = gpio_open(8);
		gpio_set_direction(trigger, GPIO_OUT);
	}
	else
	{
		gpio trigger = NULL;
	}
	for(unsigned int i = 0; i < nb_12_16; ++i) //configuration of the DMA for the filters
	{
		dma_receiv_start(filters_12_16[i]);
		dma_send_start(filters_12_16[i]);
	}
	for(unsigned int i = 0; i < nb_lp_30; ++i)
	{
		dma_receiv_start(filters_lp_35[i]);
		dma_send_start(filters_lp_35[i]);
	}


	while(1)
	{
		if(pos<l)
		{
			while(gpio_read(DRDY) == 1);//wait for the signal to be ready (when DRDY = 0, the signal can be read)
			for(j = 0;j<9;++j){//read a 24 bits messages then the 24 bits values of the 8 electrodes
				my_spi_transfer_faster(device,NULL, buf_read, 3);//transfert empty message to received the signal from the ADS1299
				if(j <= nb_ecg+nb_eeg && j != 0){//get the signal from the electrodes we are interested in
					res = (buf_read[0]<<16)|(buf_read[1]<<8)|(buf_read[2]); //reform the value with the 3 bytes received
					if(res >= 0x800000 && (res & 0x800000) != 0) //the signal is in A2 complement, the value is converted to decimal
					{
						res -= 0x1000000;
					}
					active_buff[j-1][0][pos] = res; //write the raw value in a buffer shared with python code
					nb_acq = 1;
					if(pos%2 == 0) //only take one sample out of 2 because we go from 500 Hz to 250 Hz (500 Hz is for post analysis, but computation is faster with 250 Hz)
					{
						if(j-1 < nb_12_16)
						{
							//if activated, send the raw signal to the 12-16Hz bandpass filter
							*shared_buff1 = res;
							dma_send_transfert(filters_12_16[j-1], (u32)shared_buff1, sizeof(s32));
							dma_receiv_transfert(filters_12_16[j-1], (u32)shared_buff2, sizeof(s32));
							dma_send_wait(filters_12_16[j-1]);
							dma_receiv_wait(filters_12_16[j-1]); //a implémenter pour plusieurs filtres
							res_bp1216 = *shared_buff2;
							//start standardization
							if(total_pos == 0)
							{
								moving_average_bp1216 = res_bp1216;
							}
							else
							{
								delta_bp1216 = res_bp1216 - moving_average_bp1216;
								moving_average_bp1216 = moving_average_bp1216 + ALPHA_STANDARDIZATION*delta_bp1216;
								moving_variance_bp1216 = (1-ALPHA_STANDARDIZATION)*(moving_variance_bp1216+ALPHA_STANDARDIZATION*delta_bp1216*delta_bp1216);
								moving_std_bp1216 = sqrt(moving_variance_bp1216);
								res_bp1216 = (res_bp1216-moving_average_bp1216)/(moving_std_bp1216+EPSILON);
							}
							//get the envelope
							res_envelope = res_bp1216*res_bp1216;
							if(total_pos == 0)
							{
								moving_average_envelope = res_envelope;
							}
							else
							{
								delta_envelope = res_envelope - moving_average_envelope;
								moving_average_envelope = moving_average_envelope + ALPHA_ENVELOPE*delta_envelope;
								res_envelope = moving_average_envelope;
							}
							//write the result in the buffer
							active_buff[j-1][nb_acq][pos/2] = res_envelope;
							moving_window_input2[moving_idx] = res_envelope;
							//		moving_window_input2[moving_idx] = 0; //for debugging

							nb_acq++;
						}
						if(j-1 < nb_lp_30)
						{
							//same with 30 Hz filter
							*shared_buff1 = res;
							dma_send_transfert(filters_lp_35[j-1], (u32)shared_buff1, sizeof(s32));
							dma_receiv_transfert(filters_lp_35[j-1], (u32)shared_buff2, sizeof(s32));
							dma_send_wait(filters_lp_35[j-1]);
							dma_receiv_wait(filters_lp_35[j-1]);
							res_lp35 = *shared_buff2;
							//notch filter, coeff are choosen depeding on the geographical area (selected in pythonà
							float denAccum;

							denAccum = (res_lp35 - notch_coeff1 *
									DigitalFilter_states[0]) - notch_coeff2 *
											DigitalFilter_states[1];

							res_lp35 = (notch_coeff3 * denAccum + notch_coeff4 *
									DigitalFilter_states[0]) +
											notch_coeff5 * DigitalFilter_states[1];

							DigitalFilter_states[1] = DigitalFilter_states[0];
							DigitalFilter_states[0] = denAccum;
							//standardization
							if(total_pos == 0)
							{
								moving_average_lp35 = res_lp35;
							}
							else
							{
								delta_lp35 = res_lp35 - moving_average_lp35;
								moving_average_lp35 = moving_average_lp35 + ALPHA_AV_LP*delta_lp35;
								moving_variance_lp35 = (1-ALPHA_STANDARDIZATION)*(moving_variance_lp35+ALPHA_STANDARDIZATION*delta_lp35*delta_lp35);
								moving_std_lp35 = sqrt(moving_variance_lp35);
								res_lp35 = (res_lp35-moving_average_lp35)/(moving_std_lp35+EPSILON);
							}
							//send the result back to python (not directly, but put it in the shared buffer)
							active_buff[j-1][nb_acq][pos/2] = res_lp35;
							moving_window_input1[moving_idx] = res_lp35;
						//	moving_window_input1[moving_idx] = 1; //for debugging
							/*		if(active_buff[j-1][nb_acq][pos/2] > 1.5) //threshold detector
							{
								last_audio = 0;
								send_stimulation(trigger);
							}*/
							nb_acq++;
						}
						moving_idx = (moving_idx+1)%WINDOW_SIZE;

						timestamp = 0;//result from NN when available, used for debugging
						if(nn_res_received == 0 && dma_send_idle(NN_DMA_BASE_ADDR) && dma_receiv_idle(NN_DMA_BASE_ADDR))
						{
							//when NN send back result
							nn_res_received = 1;
							timestamp = *(nn_output_buff); //write in it timestamp

							//send stimulation if possible
							if (*nn_output_buff>THRESHOLD && wait_stim == 0 && in_spindle == 0)
							{
								if (send_stimulation(TRIGGER_STIMULATION) == 1)
								{
									wait_stim = WAIT_BEFORE_STIM;
									in_spindle = WAIT_BEFORE_STIM;
								}
							}
							if (*nn_output_buff>THRESHOLD && in_spindle > 0)
							{
								in_spindle = WAIT_BEFORE_STIM;
							}
							//copy hidden vector for virtual parallelization
							for(int i = 0; i < HIDDEN_VECTOR_SIZE; ++i)
							{
								//	*(nn_input_buff+i+NB_INPUT*WINDOW_SIZE) = *(nn_output_buff+i+1);
								hidden_vector[idx_sample][i] = *(nn_output_buff+i+1);
							}
							//select the next parallel network (with it hidden vector)
							idx_sample = (idx_sample+1)%NB_PARALLELE;
						}
						for(int i = 0; i < NB_PARALLELE; ++i)
						{
							pos_sample[i]++; //increase waiting time for each network (start at 0, when reaching the time dilation value, the network can be used again)
						}
						if(nn_res_received != 0 && pos_sample[idx_sample]>=NETWORK_SAMPLING)
						{
							pos_sample[idx_sample] = 0;
							timestamp = 1000 + timestamp; //indicate that an inference has been launched
							//send the data to the network (input + hidden vector), must be rework for 2 input network
							for(int i = 0; i < WINDOW_SIZE; ++i)
							{
								*(nn_input_buff+i) = moving_window_input1[(moving_idx+i)%WINDOW_SIZE];
							}
							if (NB_INPUT>1)
							{
								for(int i = 0; i < WINDOW_SIZE; ++i)
								{
									*(nn_input_buff+i+WINDOW_SIZE) = moving_window_input2[(moving_idx+i)%WINDOW_SIZE];
								}
							}
							for(int i = 0; i < HIDDEN_VECTOR_SIZE; ++i)
							{
								*(nn_input_buff+i+WINDOW_SIZE*NB_INPUT) = hidden_vector[idx_sample][i];
//								timestamp = hidden_vector[idx_sample][0];
							}
							/*if (NB_INPUT>1)
							{
								for(int i = 0; i < HIDDEN_VECTOR_SIZE; ++i)
								{
							 *(nn_input_buff+i+WINDOW_SIZE*NB_INPUT+HIDDEN_VECTOR_SIZE) = hidden_vector2[idx_sample][i];
								}
							}*/

							// send the buffer through DMA
							dma_send_transfert(NN_DMA_BASE_ADDR, (u32)nn_input_buff, (NN_INPUT_SIZE)*sizeof(float));
							dma_receiv_transfert(NN_DMA_BASE_ADDR, (u32)nn_output_buff, (NN_OUTPUT_SIZE)*sizeof(float));
							nn_res_received = 0; // waiting for the network to finish


							//send_stimulation(trigger); // for testing audio, play the audio at the network sampling rate
						}
						if (wait_stim > 0)
						{
							--wait_stim;//decrease stimulation related counter
						}
						if (in_spindle > 0)
						{
							--in_spindle;
						}
						total_pos++;
						if(j-1 < nb_eeg)
						{
							active_buff[j-1][nb_acq][pos/2] = timestamp;//write timestamp inside python buffer
							nb_acq++;
						}

					}
				}
			}
			++last_audio;
			++pos;
		}
		if(pos == l)
		{
			//when enough data has been written inside the buffer, indicate to python to read them and start using the second buffer
			value = (value+1)%nb_buffer;
			swap_buffer(value);
			Xil_Out32(XPAR_IOP_ARDUINO_INTR_BASEADDR,0x1);//envoie de l'interruption
			Xil_Out32(XPAR_IOP_ARDUINO_INTR_BASEADDR,0x0);
			if(++cycle == nb_cycles)
			{
				break;
			}
		}
	}
	for(unsigned int i = 0; i < nb_12_16; ++i) //indicate to the FIR to stop working
	{
		dma_receiv_stop(filters_12_16[i]);
		dma_send_stop(filters_12_16[i]);
	}
	for(unsigned int i = 0; i < nb_lp_30; ++i)
	{
		dma_receiv_stop(filters_lp_35[i]);
		dma_send_stop(filters_lp_35[i]);
	}

}

void dma_send_start(int offset)
{
	if(dma_send_running(offset))
	{
		return;
	}
	Xil_Out32(offset, 0x0001); //start DMA send_channel
	while(!dma_send_running(offset));
}
void dma_receiv_start(int offset)
{
	if(dma_receiv_running(offset))
	{
		return;
	}
	Xil_Out32(offset+0x30, 0x0001); //start DMA receiv_channel
	while(!dma_receiv_running(offset));
}
void dma_send_stop(int offset)
{
	if(!dma_send_running(offset))
	{
		return;
	}
	Xil_Out32(offset, 0x0000);
	while(dma_send_running(offset));
}
void dma_receiv_stop(int offset)
{
	if(!dma_receiv_running(offset))
	{
		return;
	}
	Xil_Out32(offset+0x30, 0x0000);
	while(dma_receiv_running(offset));
}


int dma_send_idle(int offset)
{
	return (Xil_In32(offset+4) & 0x02) == 0x02;
}
int dma_receiv_idle(int offset)
{
	return (Xil_In32(offset+0x30+4) & 0x02) == 0x02;
}

void dma_send_transfert(int offset, int buff_addr, int length)
{
	Xil_Out32(offset+0x18, buff_addr);
	Xil_Out32(offset+0x28, length);
}
void dma_receiv_transfert(int offset, int buff_addr, int length)
{
	Xil_Out32(offset+0x30+0x18, buff_addr);
	Xil_Out32(offset+0x30+0x28, length);
}
void dma_send_wait(int offset)
{
	while(!dma_send_idle(offset));
}
void dma_receiv_wait(int offset)
{
	while(!dma_receiv_idle(offset));
}
int dma_send_running(int offset)
{
	return (Xil_In32(offset+4) & 0x01) == 0x00;
}
int dma_receiv_running(int offset)
{
	return (Xil_In32(offset+0x30+4) & 0x01) == 0x00;
}

int send_stimulation(gpio trigger)
{
	if (TRIGGER_STIMULATION == 0)
	{
		static int first_audio = 1;
		if(first_audio)
		{
			first_audio = 0;
		}
		else if (!dma_send_idle(AUDIO_DMA_BASE_ADDR))
		{
			return 0; //no sound send if the previous one is not over
		}
		dma_send_transfert(AUDIO_DMA_BASE_ADDR, (u32)audio_buff, size_audio_buff*sizeof(u32));
	}
	else
	{
		gpio_write(trigger, 1);
		for(int j;j<20;++j);//duration of the trigger, should be improved
		gpio_write(trigger, 0);
	}
	return 1;
}

int main(void)
{
	u32 cmd;
	spi dev;
	int l;
	u32 nb_cycles;
	u32 notch_filter_mode;
	u32 nb_buffer;
	u32 nb_ecg;
	u32 nb_eeg;
	u32 nb_12_16;
	u32 nb_lp_30;
	int id = 0;
	int nb_acq;
	dma_send_start(AUDIO_DMA_BASE_ADDR);
	dma_send_start(NN_DMA_BASE_ADDR);
	dma_receiv_start(NN_DMA_BASE_ADDR);

	Xil_Out32(XPAR_IOP_ARDUINO_INTR_BASEADDR+4,0x0);//activation interruptions
	Xil_Out32(XPAR_IOP_ARDUINO_INTR_BASEADDR,0x0);

	while(1)
	{
		while((MAILBOX_CMD_ADDR & 0x01)==0); //communication mailbox with python code
		cmd = MAILBOX_CMD_ADDR;

		switch(cmd){
		case SETUP_DEVICE: //received setup message
			dev = setup_my_device();
			MAILBOX_CMD_ADDR = 0x0;
			break;
		case READ_REGISTERS: //received read register message
			read_register(dev);
			MAILBOX_CMD_ADDR = 0x0;
			break;
		case START_ACQUIRE: //received start acquire message
			id = 0;
			nb_ecg = MAILBOX_DATA(id++);//read every information received
			nb_eeg = MAILBOX_DATA(id++);
			l = MAILBOX_DATA(id++);
			nb_cycles = MAILBOX_DATA(id++);
			notch_filter_mode = MAILBOX_DATA(id++);
			switch (notch_filter_mode)
			{
			case 1 : // notch filter 50 Hz
			{
				notch_coeff1 = -0.61410695998423581;
				notch_coeff2 =  0.98729186796473023;
				notch_coeff3 = 0.99364593398236511;
				notch_coeff4 = -0.61410695998423581;
				notch_coeff5 = 0.99364593398236511;
				break;
			}
			default: //notch filter 60 Hz
			{
				notch_coeff1 = -0.12478308884588535;
				notch_coeff2 = 0.98729186796473023;
				notch_coeff3 = 0.99364593398236511;
				notch_coeff4 = -0.12478308884588535;
				notch_coeff5 = 0.99364593398236511;
				break;
			}
			}
			nb_buffer = MAILBOX_DATA(id++);
			shared_buff1 = (s32*)(MAILBOX_DATA(id++) | 0x20000000);//si ça ne fonctionne pas, faire un "real buffer" à coté
			shared_buff2 = (s32*)(MAILBOX_DATA(id++) | 0x20000000);
			nb_12_16 = MAILBOX_DATA(id++);
			nb_lp_30 = MAILBOX_DATA(id++);
			audio_buff = (s32*)(MAILBOX_DATA(id++)); //ne fonctionne pas si | 0x20000000
			size_audio_buff = MAILBOX_DATA(id++);
			nn_input_buff = (float*)(MAILBOX_DATA(id++) | 0x20000000);
			nn_output_buff = (float*)(MAILBOX_DATA(id++) | 0x20000000);
			buffList = malloc(nb_buffer*sizeof(int*));
			for(int j = 0; j < nb_buffer; ++j)
			{
				buffList[j] = malloc((nb_ecg+nb_eeg)*sizeof(int*));
				for(int i = 0; i < (nb_ecg+nb_eeg); ++i)
				{
					nb_acq = 1;
					if(i < nb_12_16)
					{
						nb_acq++;
					}
					if(i < nb_lp_30)
					{
						nb_acq++;
					}
					if(i < nb_eeg)
					{
						nb_acq++;
					}
					buffList[j][i] = malloc(nb_acq*sizeof(int*));
					buffList[j][i][0] = (float *)(MAILBOX_DATA(id++)| 0x20000000);
					nb_acq = 1;
					if(i < nb_12_16)
					{
						buffList[j][i][nb_acq] = (float *)(MAILBOX_DATA(id++)| 0x20000000);
						nb_acq++;
					}
					if(i < nb_lp_30)
					{
						buffList[j][i][nb_acq] = (float *)(MAILBOX_DATA(id++)| 0x20000000);
						nb_acq++;
					}
					if(i < nb_eeg)
					{
						buffList[j][i][nb_acq] = (float *)(MAILBOX_DATA(id++)| 0x20000000); //timestamp buffer
						nb_acq++;
					}
				}
			}
			active_buff = buffList[0];

			start_acquire(l, nb_cycles,dev, nb_buffer, nb_ecg, nb_eeg, nb_12_16, nb_lp_30);
			MAILBOX_CMD_ADDR = 0x0;
			break;

			default:
				MAILBOX_CMD_ADDR = 0x0;
				break;
		}
	}
	return 0;
}


/*interruption
 *                 Xil_Out32(XPAR_IOP_ARDUINO_INTR_BASEADDR,0x1);
                Xil_Out32(XPAR_IOP_ARDUINO_INTR_BASEADDR,0x0);
 *
 */
