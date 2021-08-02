#include "myUtils.h"

// Constants
/*
#define GRU1_INPUT_SIZE POOL3_TOT_OUTPUT
#define GRU1_HIDDEN_SIZE 7
#define GRU1_WI_SIZE 3*GRU1_HIDDEN_SIZE*GRU1_INPUT_SIZE
#define GRU1_WH_SIZE 3*GRU1_HIDDEN_SIZE*GRU1_HIDDEN_SIZE
#define GRU1_B_SIZE 3*GRU1_HIDDEN_SIZE

void myGru1(stream_fixed_type input[GRU1_INPUT_SIZE], const parameter_fixed_type w_ih[GRU1_WI_SIZE], const parameter_fixed_type w_hh[GRU1_WH_SIZE], const parameter_fixed_type b_ih[GRU1_B_SIZE], const parameter_fixed_type b_hh[GRU1_B_SIZE], stream_fixed_type hidden_in[GRU1_HIDDEN_SIZE], stream_fixed_type output[GRU1_HIDDEN_SIZE]);
 */
#ifndef GRU_H
#define GRU_H

template <int i_size,int h_size>
class myGru{
public:
#define wi_size 3*h_size*i_size
#define wh_size 3*h_size*h_size
#define b_size 3*h_size

	void run(stream_fixed_type input[i_size], const parameter_fixed_type w_ih[wi_size], const parameter_fixed_type w_hh[wh_size], const parameter_fixed_type b_ih[b_size], const parameter_fixed_type b_hh[b_size], stream_fixed_type hidden_in[h_size], stream_fixed_type output[h_size])
	{
		// Variables
		int i, j, k, counter_input, counter_hidden;
		stream_fixed_type mult_temp,mult_temp2;
		stream_fixed_type rt[h_size], zt[h_size], nt[h_size];

		counter_input = 0;
		counter_hidden = 0;
		for (k = 0; k < 3; k++)
		{
			for (i = 0; i < h_size; i++) {
				mult_temp = 0; // Make sure the values are zero initially
				mult_temp2 = 0; // Make sure the values are zero initially
				for (j = 0; j < i_size; j++) {
					mult_temp += (w_ih[counter_input] * input[j]);
					counter_input++;
				}
				for (j = 0; j < h_size; j++) {
					mult_temp2 += (w_hh[counter_hidden] * hidden_in[j]);
					counter_hidden++;
				}
				mult_temp += b_ih[i+k*h_size];
				mult_temp2 += b_hh[i+k*h_size];
				switch (k)
				{
				case 0:
					rt[i] = sigmoid(mult_temp + mult_temp2);
					break;
				case 1:
					zt[i] = sigmoid(mult_temp+mult_temp2);
					break;
				case 2:
					nt[i] = mytanh(stream_fixed_type(mult_temp + rt[i]*mult_temp2));
					output[i] = (1-zt[i])*nt[i] + zt[i]*hidden_in[i];
					break;
				default:
					break;
				}
			}
		}
	}
};
#endif
