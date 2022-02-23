"""
Main script for PMBO

Possible arguments:
--server -> launches the server
--worker -> launches a worker
--meta -> launches the meta learner
--output_file -> name of the log file (string)
--ip_server -> ip of the server machine (string)

"""

import logging
import socket
import time
from argparse import ArgumentParser
from copy import deepcopy
from threading import Lock, Thread

import torch
from pyinstrument import Profiler
from requests import get

from pareto_network_server_utils import Server, RECV_TIMEOUT_META_FROM_SERVER, SOCKET_TIMEOUT_CONNECT_META, PORT_META, RECV_TIMEOUT_WORKER_FROM_SERVER, \
    PORT_WORKER, SOCKET_TIMEOUT_CONNECT_WORKER, ACK_TIMEOUT_WORKER_TO_SERVER, IP_SERVER, ACK_TIMEOUT_META_TO_SERVER, select_and_send_or_close_socket, poll_and_recv_or_close_socket, get_connected_socket, LOOP_SLEEP_TIME_META, LOOP_SLEEP_TIME_WORKER, LOOP_SLEEP_TIME, SEND_ALIVE
from pareto_search import LoggerWandbPareto, RUN_NAME, SurrogateModel, META_MODEL_DEVICE, train_surrogate, update_pareto, nb_parameters, MAX_NB_PARAMETERS, NB_SAMPLED_MODELS_PER_ITERATION, exp_max_pareto_efficiency, run, \
    load_network_files, dump_network_files, transform_config_dict_to_input, WANDB_PROJECT_PARETO, PARETO_ID
from portiloop_software.portiloop_python.ANN.training_experiment import initialize_dataset_config, initialize_exp_config
from portiloop_software.portiloop_python.Utils.utils import same_config_dict, sample_config_dict, MIN_NB_PARAMETERS, MAXIMIZE_F1_SCORE, PROFILE_META


# META LEARNER: ==========================================

class MetaLearner:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """

    def __init__(self, server_ip=None):
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.recv_timeout = RECV_TIMEOUT_META_FROM_SERVER
        self.__results_lock = Lock()
        self.__results = []
        self.__to_launch_lock = Lock()
        self.__to_launch = []
        self.__must_launch = False
        self.__must_launch_lock = Lock()

        logging.debug(f"local IP: {self.local_ip}")
        logging.debug(f"public IP: {self.public_ip}")
        logging.debug(f"server IP: {self.server_ip}")

        Thread(target=self.__run_thread, args=(),
               kwargs={}, daemon=True).start()
        self.run()

    def __run_thread(self):
        """
        Meta interface thread
        """
        while True:  # main client loop
            alive_time = time.time()
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(
                SOCKET_TIMEOUT_CONNECT_META, self.server_ip, PORT_META)
            if s is None:
                logging.debug("get_connected_socket failed in Meta thread")
                continue
            while True:
                # send weights
                # WEIGHTS LOCK...........................................................
                self.__to_launch_lock.acquire()
                if len(self.__to_launch) > 0:  # new experiments to send
                    if not wait_ack:
                        obj = self.__to_launch.pop(0)
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__to_launch_lock.release()
                            logging.debug(
                                "select_and_send_or_close_socket failed in Meta")
                            break
                    else:
                        elapsed = time.time() - ack_time
                        logging.debug(
                            f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_META_TO_SERVER:
                            logging.debug(
                                "INFO: ACK timed-out, breaking connection")
                            self.__to_launch_lock.release()
                            break
                else:
                    elapsed = time.time() - alive_time
                    if elapsed >= SEND_ALIVE:
                        alive_time = time.time()
                        if not wait_ack:
                            obj = "ALIVE"
                            if select_and_send_or_close_socket(obj, s):
                                ack_time = time.time()
                                wait_ack = True
                            else:
                                self.__to_launch_lock.release()
                                logging.debug(
                                    "select_and_send_or_close_socket failed in Meta")
                                break
                        else:
                            elapsed = time.time() - ack_time
                            logging.debug(
                                f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                            if elapsed >= ACK_TIMEOUT_META_TO_SERVER:
                                logging.debug(
                                    "INFO: ACK timed-out, breaking connection")
                                self.__to_launch_lock.release()
                                break
                # END LOCK.......................................................
                self.__to_launch_lock.release()
                # checks for samples batch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    logging.debug("poll failed in Meta thread")
                    break
                elif obj is not None and obj != 'ACK':  # received finished
                    logging.debug(f"DEBUG INFO: Meta interface received obj")
                    recv_time = time.time()
                    # LOCK.........................................................
                    self.__results_lock.acquire()
                    self.__results += obj
                    # END LOCK.....................................................
                    self.__results_lock.release()
                    self.__must_launch_lock.acquire()
                    self.__must_launch = True
                    self.__must_launch_lock.release()
                elif obj == 'ACK':
                    recv_time = time.time()
                    wait_ack = False
                    logging.debug(
                        f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_timeout:
                    logging.debug(
                        f"Timeout in Meta, not received anything for too long")
                    break
                time.sleep(LOOP_SLEEP_TIME_META)
            s.close()

    def run(self):
        """
        Meta learner main loop
        """
        logger = LoggerWandbPareto(RUN_NAME)
        finished_experiments, pareto_front = load_network_files()
        launched_experiments = []

        if finished_experiments is None:
            logging.debug(f"no meta dataset found, starting new run")
            finished_experiments = []  # list of dictionaries
            pareto_front = []  # list of dictionaries, subset of finished_experiments
            meta_model = SurrogateModel()
            meta_model.to(META_MODEL_DEVICE)
        else:
            logging.debug(f"existing meta dataset loaded")
            logging.debug("training new surrogate model...")
            meta_model = SurrogateModel()
            meta_model.to(META_MODEL_DEVICE)
            meta_model.train()
            meta_model, meta_loss = train_surrogate(
                meta_model, deepcopy(finished_experiments))
            logging.debug(f"surrogate model loss: {meta_loss}")

        # main meta-learning procedure:
        prev_exp = {}

        while True:
            self.__must_launch_lock.acquire()
            if self.__must_launch:
                self.__must_launch = False
                self.__must_launch_lock.release()

                if PROFILE_META:
                    pro = Profiler()
                    pro.start()

                self.__results_lock.acquire()
                temp_results = deepcopy(self.__results)
                self.__results = []
                self.__results_lock.release()
                for res in temp_results:
                    if 'best_epoch' in res.keys():
                        logging.debug(
                            f"best epoch for the model received : {res['best_epoch']}")
                    to_remove = -1
                    to_update = -1
                    for i, exp in enumerate(launched_experiments):
                        if same_config_dict(exp["config_dict"], res["config_dict"]):
                            to_remove = i
                            break
                    for i, exp in enumerate(finished_experiments):
                        if same_config_dict(exp["config_dict"], res["config_dict"]):
                            to_update = i
                            break

                    if to_remove >= 0:
                        launched_experiments.pop(to_remove)
                    if to_update >= 0:
                        finished_experiments[to_update]["software_cost"] = min(
                            finished_experiments[to_update]["software_cost"], res["software_cost"])
                        pareto_front = update_pareto(
                            finished_experiments[to_update]["software_cost"], pareto_front)
                    else:
                        pareto_front = update_pareto(res, pareto_front)
                        finished_experiments.append(res)
                    dump_network_files(finished_experiments, pareto_front)
                    prev_exp = res
                # train before sampling a new model
                if len(finished_experiments) > 0 and prev_exp != {}:
                    logging.debug("training new surrogate model...")

                    meta_model = SurrogateModel()
                    meta_model.to(META_MODEL_DEVICE)

                    meta_model.train()
                    meta_model, meta_loss = train_surrogate(
                        meta_model, deepcopy(finished_experiments))

                    logging.debug(f"surrogate model loss: {meta_loss}")

                    logger.log(surrogate_loss=meta_loss,
                               surprise=prev_exp["surprise"], all_experiments=finished_experiments, pareto_front=pareto_front)

                num_experiment = len(finished_experiments) + \
                    len(launched_experiments)
                logging.debug("---")
                logging.debug(f"ITERATION N° {num_experiment}")

                exp = {}
                exps = []
                model_selected = False
                meta_model.eval()

                while not model_selected:
                    exp = {}

                    # sample model
                    config_dict, unrounded = sample_config_dict(name=RUN_NAME + "_" + str(
                        num_experiment), previous_exp=prev_exp, all_exp=finished_experiments + launched_experiments + exps)

                    nb_params = nb_parameters(config_dict)
                    if nb_params > MAX_NB_PARAMETERS or nb_params < MIN_NB_PARAMETERS:
                        continue

                    with torch.no_grad():
                        input = transform_config_dict_to_input(config_dict)
                        predicted_cost = meta_model(input).item()

                    exp["cost_hardware"] = nb_params
                    exp["cost_software"] = predicted_cost
                    exp["config_dict"] = config_dict
                    exp["unrounded"] = unrounded

                    exps.append(exp)

                    if len(exps) >= NB_SAMPLED_MODELS_PER_ITERATION:
                        # select model
                        model_selected = True
                        exp = exp_max_pareto_efficiency(
                            exps, pareto_front, finished_experiments)

                logging.debug(f"config: {exp['config_dict']}")
                logging.debug(f"nb parameters: {exp['cost_hardware']}")
                logging.debug(f"predicted cost: {exp['cost_software']}")

                self.__to_launch_lock.acquire()
                self.__to_launch.append(exp)
                self.__to_launch_lock.release()
                launched_experiments.append(exp)
                prev_exp = {}

                if PROFILE_META:
                    pro.stop()
                    logging.debug(pro.output_text(unicode=False, color=False))

            else:
                self.__must_launch_lock.release()
            time.sleep(LOOP_SLEEP_TIME)


# WORKER: ===================================


class Worker:
    def __init__(self, data_config, server_ip=None):

        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.recv_timeout = RECV_TIMEOUT_WORKER_FROM_SERVER

        self.__finished_exp = None
        self.__finished_exp_lock = Lock()
        self.__exp_to_run = None
        self.__exp_to_run_lock = Lock()

        self.data_config = data_config
        self.exp_config = initialize_exp_config()

        logging.debug(f"local IP: {self.local_ip}")
        logging.debug(f"public IP: {self.public_ip}")
        logging.debug(f"server IP: {self.server_ip}")

        Thread(target=self.__run_thread, args=(),
               kwargs={}, daemon=True).start()
        self.run()

    def __run_thread(self):
        """
        Worker thread
        """
        while True:  # main client loop
            alive_time = time.time()
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(
                SOCKET_TIMEOUT_CONNECT_WORKER, self.server_ip, PORT_WORKER)
            if s is None:
                logging.debug("get_connected_socket failed in worker")
                continue
            while True:
                # send buffer
                # BUFFER LOCK.............................................................
                self.__finished_exp_lock.acquire()
                if self.__finished_exp is not None:  # a new result is available
                    logging.debug("new result available")
                    if not wait_ack:
                        obj = deepcopy(self.__finished_exp)
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__finished_exp_lock.release()
                            logging.debug(
                                "select_and_send_or_close_socket failed in worker")
                            break
                        self.__finished_exp = None
                    else:
                        elapsed = time.time() - ack_time
                        logging.debug(
                            f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_WORKER_TO_SERVER:
                            logging.debug(
                                "INFO: ACK timed-out, breaking connection")
                            self.__finished_exp_lock.release()
                            break
                else:
                    elapsed = time.time() - alive_time
                    if elapsed >= SEND_ALIVE:
                        alive_time = time.time()
                        if not wait_ack:
                            obj = "ALIVE"
                            if select_and_send_or_close_socket(obj, s):
                                ack_time = time.time()
                                wait_ack = True
                            else:
                                self.__finished_exp_lock.release()
                                logging.debug(
                                    "select_and_send_or_close_socket failed in Meta")
                                break
                        else:
                            elapsed = time.time() - ack_time
                            logging.debug(
                                f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                            if elapsed >= ACK_TIMEOUT_META_TO_SERVER:
                                logging.debug(
                                    "INFO: ACK timed-out, breaking connection")
                                self.__finished_exp_lock.release()
                                break
                # END BUFFER LOCK.........................................................
                self.__finished_exp_lock.release()
                # checks for new experiments to launch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    logging.debug(f"INFO: worker poll failed")
                    break
                elif obj is not None and obj != 'ACK':
                    logging.debug(f"DEBUG INFO: worker received obj")
                    recv_time = time.time()
                    # LOCK.......................................................
                    self.__exp_to_run_lock.acquire()
                    self.__exp_to_run = obj
                    # END LOCK...................................................
                    self.__exp_to_run_lock.release()
                elif obj == 'ACK':
                    recv_time = time.time()
                    wait_ack = False
                    logging.debug(
                        f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_timeout:
                    logging.debug(
                        f"Timeout in worker, not received anything for too long")
                    break
                time.sleep(LOOP_SLEEP_TIME_WORKER)
            s.close()

    def run(self):
        while True:
            self.__exp_to_run_lock.acquire()
            if self.__exp_to_run is not None:
                exp = deepcopy(self.__exp_to_run)
                self.__exp_to_run = None
                self.__exp_to_run_lock.release()
                predicted_loss = exp['cost_software']

                # TODO: Get a data config dict and a exp dict
                logging.debug("Launch run")
                best_loss, best_f1_score, exp["best_epoch"] = run(
                    exp["config_dict"], self.data_config, self.exp_config, f"{WANDB_PROJECT_PARETO}_runs_{PARETO_ID}", save_model=False, unique_name=True)
                logging.debug("Run finished")
                exp["cost_software"] = 1 - \
                    best_f1_score if MAXIMIZE_F1_SCORE else best_loss
                exp['surprise'] = exp["cost_software"] - predicted_loss
                self.__finished_exp_lock.acquire()
                logging.debug("Before copying exp result")
                self.__finished_exp = deepcopy(exp)
                logging.debug("After copying exp result")
                self.__finished_exp_lock.release()
            else:
                self.__exp_to_run_lock.release()
            time.sleep(LOOP_SLEEP_TIME)


def main(args, data_config=None):
    if args.server:
        logging.debug("INFO: now running: server")
        Server()
    elif args.worker:
        Worker(data_config, server_ip=args.ip_server)
        logging.debug("INFO: now running: worker")
    elif args.meta:
        MetaLearner(server_ip=args.ip_server)
        logging.debug("INFO: now running: meta")
    else:
        logging.debug("ERROR: wrong argument")
    while True:
        time.sleep(10.0)
        pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--meta', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--ip_server', type=str, default=IP_SERVER)
    parser.add_argument('--dataset_path', type=str, default=None)

    args = parser.parse_args()
    if args.output_file is not None:
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            filename=args.output_file, level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

    if args.worker:
        dataset_config = initialize_dataset_config(
            dataset_path=args.dataset_path)

    main(args, data_config=dataset_config)
