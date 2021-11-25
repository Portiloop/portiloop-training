#include "myUtils.h"

stream_fixed_type sigmoid(stream_fixed_type value)
{
	stream_fixed_type x = -value;
	return ((stream_fixed_type)1)/(stream_fixed_type(stream_fixed_type(1) + hls::exp(x)));
}

//tanh function only return positive value
stream_fixed_type mytanh(stream_fixed_type value)
{
	stream_fixed_type coeff = 1;
	if(value < 0)
	{
		coeff = -1;
	}
	return coeff*hls::abs(hls::tanh(value));
}
