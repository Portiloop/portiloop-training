############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project oneInputNN
set_top oneInputNN
add_files src/oneInputNN/oneInputNN.h
add_files src/oneInputNN/oneInputNN.cpp
add_files src/oneInputNN/myUtils.h
add_files src/oneInputNN/myUtils.cpp
add_files src/oneInputNN/myMaxPool.h
add_files src/oneInputNN/myMaxPool.cpp
add_files src/oneInputNN/myGru.h
add_files src/oneInputNN/myGru.cpp
add_files src/oneInputNN/myFc.h
add_files src/oneInputNN/myFc.cpp
add_files src/oneInputNN/myConvLayer.h
add_files src/oneInputNN/myConvLayer.cpp
open_solution "solution1"
set_part {xc7z020clg400-1}
create_clock -period 2 -name default
#source "./oneInputNN/solution1/directives.tcl"
#csim_design
csynth_design
#cosim_design
export_design -rtl verilog -format ip_catalog
