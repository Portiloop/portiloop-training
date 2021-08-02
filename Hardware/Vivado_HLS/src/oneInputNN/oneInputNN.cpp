#include "oneInputNN.h"

void oneInputNN(datatype input[TOT_INPUT], datatype output[TOT_OUTPUT])
{
#pragma HLS INTERFACE ap_ctrl_none port=return // No block-level protocol
#pragma HLS INTERFACE axis port=input // Declaring I/O as stream
#pragma HLS INTERFACE axis port=output // Declaring I/O as stream

	bool last[TOT_OUTPUT] = {};
	last[TOT_OUTPUT-1] = true; // Setting the last flag HIGH for the last element

	stream_fixed_type conv1_input1_in[INPUT1];

	stream_fixed_type gru1_input1_hidden_in[GRU_HIDDEN_SIZE];

	stream_fixed_type gru1_input1_hidden_out[GRU_HIDDEN_SIZE];

	stream_fixed_type temp_output1[INPUT2*CHANNEL_SIZE]; //CONV1_TOT_OUTPUT == max output value for every layers
	stream_fixed_type temp_output2[INPUT3*CHANNEL_SIZE];


	int i;
	for (i = 0; i < INPUT1; ++i) {
#pragma HLS PIPELINE
		conv1_input1_in[i] = input[i].data;
	}
	for (i = 0; i < GRU_HIDDEN_SIZE; ++i) {
#pragma HLS PIPELINE
		gru1_input1_hidden_in[i] = input[i+IN_OFSET1].data;
	}
	conv1.run(conv1_input1_in,conv1_input1_kernel,conv1_input1_bias, temp_output1);
	pool1.run(temp_output1, temp_output2);
	conv2.run(temp_output2,conv2_input1_kernel,conv2_input1_bias, temp_output1);
	pool2.run(temp_output1, temp_output2);
	conv3.run(temp_output2,conv3_input1_kernel,conv3_input1_bias, temp_output1);
	pool3.run(temp_output1, temp_output2);
	gru1.run(temp_output2, gru1_input1_w_ih, gru1_input1_w_hh, gru1_input1_b_ih, gru1_input1_b_hh, gru1_input1_hidden_in, gru1_input1_hidden_out);

	for(i = 0; i < GRU_HIDDEN_SIZE; ++i)
	{
		temp_output1[i] = gru1_input1_hidden_out[i];
	}

	fc.run(temp_output1, fc_weight, fc_bias, temp_output2);

	// Send output to the stream
	for(i = 0; i < FC_OUTPUT_SIZE; ++i)
	{
#pragma HLS PIPELINE
		output[i].data = temp_output2[i];
		output[i].last = last[i];
	}


	for (i = 0; i < GRU_HIDDEN_SIZE; i++) {
#pragma HLS PIPELINE
		output[OUT_OFSET1+i].data = gru1_input1_hidden_out[i];
		output[OUT_OFSET1+i].last = last[OUT_OFSET1+i];
	}


}
