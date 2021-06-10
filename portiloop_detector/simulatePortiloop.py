# script that will test the network like if it was on the device
import logging


def simulate():
    fpga_nn_exec_time = 10  # equivalent to 40 ms
    error_fpga_exec_time = 3  # to be sure there is no overlap
    seq_stride = 42

    nb_parallel_runs = seq_stride // (fpga_nn_exec_time + error_fpga_exec_time)
    logging.debug(f"nb_parallel_runs: {nb_parallel_runs}")
    stride_between_runs = seq_stride // nb_parallel_runs
    logging.debug(f"stride_between_runs: {stride_between_runs}")


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    simulate()
