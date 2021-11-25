#include <ap_axi_sdata.h> // Needed for Fixed-point datatype
#include <hls_math.h>

#ifndef UTILS_H
#define UTILS_H

typedef ap_fixed<25,7> stream_fixed_type; // Fixed-point datatype of FIX_32_16
typedef ap_fixed<21,3> parameter_fixed_type; // Fixed-point parameter of FIX_8

// Structure for stream items
struct datatype {
	float data;
	bool last;
};

stream_fixed_type sigmoid(stream_fixed_type value);

stream_fixed_type mytanh(stream_fixed_type value);

#endif
