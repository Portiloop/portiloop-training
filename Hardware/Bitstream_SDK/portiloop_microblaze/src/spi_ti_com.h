/*
 * spi_ti_com.h
 *
 *  Created on: 22 oct. 2020
 *      Author: Nicolas
 */

#ifndef SRC_SPI_TI_COM_H_
#define SRC_SPI_TI_COM_H_
#include "gpio.h"
#include <circular_buffer.h>
#include "stdlib.h"
#include "tgmath.h"

#define SETUP_DEVICE       0x1
#define READ_REGISTERS      0x3
#define START_ACQUIRE      0x5

#define FILTER_LP_35_0_DMA_BASE_ADDR 0x41E00000
#define FILTER_12_16_0_DMA_BASE_ADDR 0x41E10000
#define FILTER_12_16_1_DMA_BASE_ADDR 0x41E20000

#define AUDIO_DMA_BASE_ADDR 		 0x41E30000
#define NN_DMA_BASE_ADDR 			 0x41E40000


#define ALPHA_AV_LP 0.1
#define ALPHA_STANDARDIZATION 0.001
#define ALPHA_ENVELOPE 0.01
#define EPSILON 0.000001

#define NB_INPUT 1
#define WINDOW_SIZE 54
#define HIDDEN_VECTOR_SIZE 7*1*NB_INPUT //hidden size * nb gru layer * nb input
#define NN_INPUT_SIZE NB_INPUT*WINDOW_SIZE + HIDDEN_VECTOR_SIZE
#define NN_OUTPUT_SIZE 1 + HIDDEN_VECTOR_SIZE
#define NETWORK_SAMPLING 42 // 500 = value to test the audio latency
#define WAIT_BEFORE_STIM 100
#define THRESHOLD 0.5

#define TRIGGER_STIMULATION 0

#define NB_PARALLELE 8

static u32 filters_12_16[2] = {FILTER_12_16_0_DMA_BASE_ADDR, FILTER_12_16_1_DMA_BASE_ADDR};
static u32 filters_lp_35[1] = {FILTER_LP_35_0_DMA_BASE_ADDR};

const unsigned char WAKEUP = 0b00000010;     // Wake-up from standby mode
const unsigned char STANDBY = 0b00000100;   // Enter Standby mode
const unsigned char RESET = 0b00000110;   // Reset the device
const unsigned char START = 0b00001000;   // Start and restart (synchronize) conversions
const unsigned char STOP = 0b00001010;   // Stop conversion
const unsigned char RDATAC = 0b00010000;   // Enable Read Data Continuous mode (default mode at power-up)
const unsigned char SDATAC = 0b00010001;   // Stop Read Data Continuous mode
const unsigned char RDATA = 0b00010010;   // Read data by command; supports multiple read back

//Register Read Commands
const unsigned char RREG = 0b00100000;
const unsigned char WREG = 0b01000000;

static float **** buffList;
static float *** active_buff;
static int pos;
volatile s32 * shared_buff1;
volatile s32 * shared_buff2; //maybe replace with float?
volatile s32 * audio_buff;
volatile float * nn_input_buff;
volatile float * nn_output_buff;
static int size_audio_buff;

static float notch_coeff1;
static float notch_coeff2;
static float notch_coeff3;
static float notch_coeff4;
static float notch_coeff5;

void dma_send_start(int offset);
void dma_receiv_start(int offset);
int dma_send_idle(int offset);
int dma_receiv_idle(int offset);
void dma_send_transfert(int offset, int buff_addr, int length);
void dma_receiv_transfert(int offset, volatile int buff_addr, int length);
void dma_send_wait(int offset);
void dma_receiv_wait(int offset);
int dma_send_running(int offset);
int dma_receiv_running(int offset);
void dma_send_stop(int offset);
void dma_receiv_stop(int offset);

int send_stimulation(gpio);

spi setup_spi();
spi configure_reg(spi device);
spi setup_my_device();
void read_register(spi device);
void swap_buffer(int id);
void start_acquire(int l, int nb_cycles, spi device, int nb_buffer, int nb_ecg,int nb_eeg,int nb_12_16,int nb_05_35);

#endif /* SRC_SPI_TI_COM_H_ */
