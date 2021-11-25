#include "myUtils.h"

// Constants Editable
/*
#define POOL1_K_SIZE 7
#define POOL1_PADDING 0 //int((POOL1_K_SIZE-1)/2)
#define POOL1_STRIDE 1
#define POOL1_DILATION 1
#define POOL1_C_SIZE CONV1_C_OUT_SIZE
#define POOL1_SIZE CONV1_OUTPUT_SIZE


// Constants auto computed

#define POOL1_TOT_INPUT POOL1_C_SIZE*POOL1_SIZE
#define POOL1_OUTPUT_SIZE int((POOL1_SIZE + 2*POOL1_PADDING - POOL1_DILATION*(POOL1_K_SIZE-1) - 1)/POOL1_STRIDE + 1)
#define POOL1_TOT_OUTPUT POOL1_C_SIZE*POOL1_OUTPUT_SIZE

void myMaxPool1(stream_fixed_type input[POOL1_TOT_INPUT], stream_fixed_type output[POOL1_TOT_OUTPUT]);

 */
#ifndef POOL_H
#define POOL_H

template <int i_size,int k_size,int c_size,int stride,int dilation>
class myMaxPool{
public:
#define padding 0 //fixed to 0 because it's not implemented anymore
#define o_size int((i_size + 2*padding - dilation*(k_size-1) - 1)/stride + 1)

	void run(stream_fixed_type input[i_size*c_size], stream_fixed_type output[o_size*c_size])
	{
		// Variables
		int i, j, k;
		//	stream_fixed_type x[POOL1_C_SIZE][POOL1_SIZE+2*POOL1_PADDING];
		stream_fixed_type max_temp;

		for (i = 0; i < c_size; ++i) {
			for (j = 0; j < o_size; ++j) {
				max_temp = input[i*i_size + j*stride];
				for(k = 1; k < k_size; ++k)
				{
					if(input[i*i_size + j*stride+k] > max_temp)
					{
						max_temp = input[i*i_size + j*stride+k*dilation];
					}
				}
				output[i*o_size+j] = max_temp;
			}
		}
	}
};
#endif
