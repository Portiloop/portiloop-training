#include "myUtils.h"

// Constants
/*
#define FC_INPUT_SIZE GRU1_HIDDEN_SIZE
#define FC_OUTPUT_SIZE 1
#define FC_WEIGHT_SIZE FC_INPUT_SIZE*FC_OUTPUT_SIZE

void myFc(stream_fixed_type input[FC_INPUT_SIZE], const parameter_fixed_type weight[FC_WEIGHT_SIZE], const parameter_fixed_type bias[FC_OUTPUT_SIZE], stream_fixed_type output[FC_OUTPUT_SIZE]);
 */

#ifndef FC_H
#define FC_H

template <int i_size, int o_size>
class myFc{
public:
#define weight_size i_size*o_size

	void run(stream_fixed_type input[i_size], const parameter_fixed_type weight[weight_size], const parameter_fixed_type bias[o_size], stream_fixed_type output[o_size])
	{
		// Variables
		int i, j, k, l;
		stream_fixed_type mult_temp;

		for(i = 0; i < o_size; ++i)
		{
			mult_temp = 0;
			for(j = 0; j<i_size; ++j)
			{
				mult_temp += (input[j]*weight[j+i*i_size]);
			}
			mult_temp += bias[i];
			output[i] = sigmoid(mult_temp);

		}
	}
};

#endif
