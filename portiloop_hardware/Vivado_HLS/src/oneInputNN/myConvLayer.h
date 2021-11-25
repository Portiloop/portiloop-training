#include "myUtils.h"
// Constants Editable
/*
#define CONV1_SIZE 54
#define CONV1_K_SIZE 7
#define CONV1_C_OUT_SIZE 31
#define CONV1_C_IN_SIZE 1
#define CONV1_PADDING 0 //int((CONV1_K_SIZE-1)/2)
#define CONV1_STRIDE 1
#define CONV1_DILATION 1

// Constants auto computed

#define CONV1_TOT_INPUT CONV1_C_IN_SIZE*CONV1_SIZE
#define CONV1_OUTPUT_SIZE int((CONV1_SIZE + 2*CONV1_PADDING - CONV1_DILATION*(CONV1_K_SIZE-1) - 1)/CONV1_STRIDE + 1)
#define CONV1_BIAS CONV1_C_OUT_SIZE
#define CONV1_TOT_OUTPUT CONV1_C_OUT_SIZE*CONV1_OUTPUT_SIZE

void myConvLayer1(stream_fixed_type input[CONV1_TOT_INPUT], const parameter_fixed_type kernel[CONV1_C_OUT_SIZE][CONV1_C_IN_SIZE][CONV1_K_SIZE], const parameter_fixed_type bias[CONV1_BIAS], stream_fixed_type output[CONV1_TOT_OUTPUT]);
*/

#ifndef CONV_H
#define CONV_H

template <int i_size,int k_size,int c_out_size,int c_in_size,int stride,int dilation>
class myConvLayer{
public:
#define padding 0 //fixed to 0 because it's not implemented anymore
#define o_size int((i_size + 2*padding - dilation*(k_size-1) - 1)/stride + 1)
	void run(stream_fixed_type input[i_size*c_in_size], const parameter_fixed_type kernel[c_out_size][c_in_size][k_size], const parameter_fixed_type bias[c_out_size], stream_fixed_type output[o_size*c_out_size])
	{
		// Variables
		int i, j, k, l;
		stream_fixed_type mult_temp;

		for (i = 0; i < c_out_size; ++i) {
			for (j = 0; j < o_size; ++j) {
				mult_temp = -1;
				mult_temp = 0;
				for(l = 0; l<c_in_size; ++l)
				{

					for(k = 0; k < k_size; ++k)
					{
						mult_temp += input[l*i_size + j*stride+k*dilation]*kernel[i][l][k];
					}
				}
				mult_temp += bias[i];
				stream_fixed_type value = 0;//RELU here because not taking any extra space
				if(mult_temp > 0)
				{
					value = mult_temp;
				}
				output[i*o_size+j] = value;
			}
		}
	}
};

#endif
