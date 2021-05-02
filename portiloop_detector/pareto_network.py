# Networking ===========================================================================

import socket
import time
from argparse import ArgumentParser
from threading import Lock, Thread

import select
import torch
from requests import get
import datetime
from copy import deepcopy
import pickle

from pareto_search import LoggerWandbPareto, load_files, RUN_NAME, SurrogateModel, META_MODEL_DEVICE, train_surrogate, same_config_dict, update_pareto, sample_config_dict, nb_parameters, MAX_NB_PARAMETERS, NB_SAMPLED_MODELS_PER_ITERATION, exp_max_pareto_efficiency, dump_files, run, \
    load_network_files, dump_network_files

IP_SERVER = "142.182.5.48"  # Yann = "45.74.221.204"; Nicolas = "142.182.5.48"
PORT_META = 6666
PORT_WORKER = 6667

WAIT_BEFORE_RECONNECTION = 500.0

SOCKET_TIMEOUT_COMMUNICATE = 3600.0

SOCKET_TIMEOUT_ACCEPT_META = 3600.0
SOCKET_TIMEOUT_ACCEPT_WORKER = 3600.0

ACK_TIMEOUT_SERVER_TO_WORKER = 60.0
ACK_TIMEOUT_SERVER_TO_META = 60.0
ACK_TIMEOUT_META_TO_SERVER = 60.0
ACK_TIMEOUT_WORKER_TO_SERVER = 60.0

RECV_TIMEOUT_WORKER_FROM_SERVER = 3600.0
RECV_TIMEOUT_META_FROM_SERVER = 3600.0

SELECT_TIMEOUT_OUTBOUND = 3600.0
SELECT_TIMEOUT_INBOUND = 3600.0

SOCKET_TIMEOUT_CONNECT_META = 3600.0
SOCKET_TIMEOUT_CONNECT_WORKER = 3600.0

LOOP_SLEEP_TIME = 1.0

LEN_QUEUE_TO_LAUNCH = 5

HEADER_SIZE = 12

BUFFER_SIZE = 4096

PRINT_BYTESIZES = True


def print_with_timestamp(s):
    x = datetime.datetime.now()
    sx = x.strftime("%x %X ")
    print(sx + str(s))


def send_ack(sock):
    return send_object(sock, None, ack=True)


def send_object(sock, obj, ack=False):
    """
    If ack, this will ignore obj and send the ACK request
    If raw, obj must be a binary string
    Call only after select on a socket with a (long enough) timeout.
    Returns True if sent successfully, False if connection lost.
    """
    if ack:
        msg = bytes(f"{'ACK':<{HEADER_SIZE}}", 'utf-8')
    else:
        msg = pickle.dumps(obj)
        msg = bytes(f"{len(msg):<{HEADER_SIZE}}", 'utf-8') + msg
        if PRINT_BYTESIZES:
            print_with_timestamp(f"Sending {len(msg)} bytes.")
    try:
        sock.sendall(msg)
    except OSError:  # connection closed or broken
        return False
    return True


def recv_object(sock):
    """
    If the request is PING or PONG, this will return 'PINGPONG'
    If the request is ACK, this will return 'ACK'
    If the request is PING, this will automatically send the PONG answer
    Call only after select on a socket with a (long enough) timeout.
    Returns the object if received successfully, None if connection lost.
    This sends the ACK request back to sock when an object transfer is complete
    """
    # first, we receive the header (inefficient but prevents collisions)
    msg = b''
    l = len(msg)
    while l != HEADER_SIZE:
        try:
            recv_msg = sock.recv(HEADER_SIZE - l)
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print_with_timestamp(f"DEBUG: l:{l}")
    # print_with_timestamp("DEBUG: data len:", msg[:HEADER_SIZE])
    # print_with_timestamp(f"DEBUG: msg[:4]: {msg[:4]}")
    if msg[:3] == b'ACK':
        return 'ACK'
    msglen = int(msg[:HEADER_SIZE])
    # print_with_timestamp(f"DEBUG: receiving {msglen} bytes")
    t_start = time.time()
    # now, we receive the actual data (no more than the data length, again to prevent collisions)
    msg = b''
    l = len(msg)
    while l != msglen:
        try:
            recv_msg = sock.recv(min(BUFFER_SIZE, msglen - l))  # this will not receive more bytes than required
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print_with_timestamp(f"DEBUG2: l:{l}")
    # print_with_timestamp("DEBUG: final data len:", l)
    # print_with_timestamp(f"DEBUG: finished receiving after {time.time() - t_start}s.")
    send_ack(sock)
    return pickle.loads(msg)


def get_listening_socket(timeout, ip_bind, port_bind):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # to reuse address on Linux
    s.bind((ip_bind, port_bind))
    s.listen(5)
    return s


def get_connected_socket(timeout, ip_connect, port_connect):
    """
    returns the connected socket
    returns None if connect failed
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip_connect, port_connect))
    except OSError:  # connection broken or timeout
        print_with_timestamp(f"INFO: connect() timed-out or failed, sleeping {WAIT_BEFORE_RECONNECTION}s")
        s.close()
        time.sleep(WAIT_BEFORE_RECONNECTION)
        return None
    s.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
    return s


def accept_or_close_socket(s):
    """
    returns conn, addr
    None None in case of failure
    """
    conn = None
    try:
        conn, addr = s.accept()
        conn.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
        return conn, addr
    except OSError:
        # print_with_timestamp(f"INFO: accept() timed-out or failed, sleeping {WAIT_BEFORE_RECONNECTION}s")
        if conn is not None:
            conn.close()
        s.close()
        time.sleep(WAIT_BEFORE_RECONNECTION)
        return None, None


def select_and_send_or_close_socket(obj, conn):
    """
    Returns True if success
    False if disconnected (closes sockets)
    """
    print_with_timestamp(f"DEBUG: start select")
    _, wl, xl = select.select([], [conn], [conn], SELECT_TIMEOUT_OUTBOUND)  # select for writing
    print_with_timestamp(f"DEBUG: end select")
    if len(xl) != 0:
        print_with_timestamp("INFO: error when writing, closing socket")
        conn.close()
        return False
    if len(wl) == 0:
        print_with_timestamp("INFO: outbound select() timed out, closing socket")
        conn.close()
        return False
    elif not send_object(conn, obj):  # error or timeout
        print_with_timestamp("INFO: send_object() failed, closing socket")
        conn.close()
        return False
    return True


def poll_and_recv_or_close_socket(conn):
    """
    Returns True, obj is success (obj is None if nothing was in the read buffer when polling)
    False, None otherwise
    """
    rl, _, xl = select.select([conn], [], [conn], 0.0)  # polling read channel
    if len(xl) != 0:
        print_with_timestamp("INFO: error when polling, closing sockets")
        conn.close()
        return False, None
    if len(rl) == 0:  # nothing in the recv buffer
        return True, None
    obj = recv_object(conn)
    if obj is None:  # socket error
        print_with_timestamp("INFO: error when receiving object, closing sockets")
        conn.close()
        return False, None
    elif obj == 'PINGPONG':
        return True, None
    else:
        # print_with_timestamp(f"DEBUG: received obj:{obj}")
        return True, obj


class Server:
    """
    This is the main server
    This lets 1 TrainerInterface and n RolloutWorkers connect
    This buffers experiences sent by RolloutWorkers
    This periodically sends the buffer to the TrainerInterface
    This also receives the weights from the TrainerInterface and broadcast them to the connected RolloutWorkers
    If trainer_on_localhost is True, the server only listens on trainer_on_localhost. Then the trainer is expected to talk on trainer_on_localhost.
    Otherwise, the server also listens to the local ip and the trainer is expected to talk on the local ip (port forwarding).
    """

    def __init__(self):
        self.__finished_lock = Lock()
        self.__finished = []
        self.__to_launch_lock = Lock()
        self.__to_launch = []
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())

        print_with_timestamp(f"INFO SERVER: local IP: {self.local_ip}")
        print_with_timestamp(f"INFO SERVER: public IP: {self.public_ip}")

        Thread(target=self.__workers_thread, args=('',), kwargs={}, daemon=True).start()
        Thread(target=self.__metas_thread, args=('',), kwargs={}, daemon=True).start()

    def __metas_thread(self, ip):
        """
        This waits for new potential Trainers to connect
        When a new Trainer connects, this instantiates a new thread to handle it
        """
        while True:  # main loop
            s = get_listening_socket(SOCKET_TIMEOUT_ACCEPT_META, ip, PORT_META)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                # print_with_timestamp("DEBUG: accept_or_close_socket failed in trainers thread")
                continue
            print_with_timestamp(f"INFO METAS THREAD: server connected by meta at address {addr}")
            Thread(target=self.__meta_thread, args=(conn,), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __meta_thread(self, conn):
        """
        This periodically sends the local buffer to the TrainerInterface (when data is available)
        When the TrainerInterface sends new weights, this broadcasts them to all connected RolloutWorkers
        """
        ack_time = time.time()
        wait_ack = False
        is_working = False
        while True:
            # send samples
            if not is_working:
                self.__to_launch_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__to_launch) < LEN_QUEUE_TO_LAUNCH:  # send request to meta
                    if not wait_ack:
                        self.__finished_lock.acquire()
                        obj = deepcopy(self.__finished)
                        self.__finished = []
                        self.__finished_lock.release()
                        if select_and_send_or_close_socket(obj, conn):
                            is_working = True
                            wait_ack = True
                            ack_time = time.time()
                        else:
                            print_with_timestamp("INFO: failed sending object to meta")
                            self.__to_launch_lock.release()
                            break
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_SERVER_TO_META:
                            print_with_timestamp("INFO: ACK timed-out, breaking connection")
                            self.__to_launch_lock.release()
                            break
                self.__to_launch_lock.release()  # END BUFFER LOCK.........................................................
            # checks for weights
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("DEBUG: poll failed in meta thread")
                break
            elif obj is not None and obj != 'ACK':
                is_working = False
                print_with_timestamp(f"DEBUG INFO: meta thread received obj")
                self.__to_launch_lock.acquire()  # LOCK.......................................................
                self.__to_launch.append(obj)
                self.__to_launch_lock.release()  # END LOCK...................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(LOOP_SLEEP_TIME)  # TODO: adapt

    def __workers_thread(self, ip):
        """
        This waits for new potential RolloutWorkers to connect
        When a new RolloutWorker connects, this instantiates a new thread to handle it
        """
        while True:  # main loop
            s = get_listening_socket(SOCKET_TIMEOUT_ACCEPT_WORKER, ip, PORT_WORKER)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                continue
            print_with_timestamp(f"INFO WORKERS THREAD: server connected by worker at address {addr}")
            Thread(target=self.__worker_thread, args=(conn,), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __worker_thread(self, conn):
        """
        Thread handling connection to a single RolloutWorker
        """
        # last_ping = time.time()
        ack_time = time.time()
        wait_ack = False
        is_working = False
        while True:
            # send weights
            if not is_working:
                self.__to_launch_lock.acquire()  # LOCK...............................................................
                if len(self.__to_launch) > 0:  # exps to be sent
                    if not wait_ack:
                        obj = self.__to_launch.pop(0)
                        if select_and_send_or_close_socket(obj, conn):
                            is_working = True
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__to_launch_lock.release()
                            print_with_timestamp("DEBUG: select_and_send_or_close_socket failed in worker thread")
                            break
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"INFO: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_SERVER_TO_WORKER:
                            print_with_timestamp("INFO: ACK timed-out, breaking connection")
                            self.__to_launch_lock.release()
                            break
                self.__to_launch_lock.release()  # END WEIGHTS LOCK...........................................................
            # checks for samples
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("DEBUG: poll failed in worker thread")
                break
            elif obj is not None and obj != 'ACK':
                is_working = False
                print_with_timestamp(f"DEBUG INFO: worker thread received obj")
                self.__finished_lock.acquire()  # BUFFER LOCK.............................................................
                self.__finished.append(obj)
                self.__finished_lock.release()  # END BUFFER LOCK.........................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(LOOP_SLEEP_TIME)


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
        self.recv_tiemout = RECV_TIMEOUT_META_FROM_SERVER
        self.__results_lock = Lock()
        self.__results = []
        self.__to_launch_lock = Lock()
        self.__to_launch = []
        self.__must_launch = False
        self.__must_launch_lock = Lock()

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"server IP: {self.server_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()
        self.run()

    def __run_thread(self):
        """
        Meta interface thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(SOCKET_TIMEOUT_CONNECT_META, self.server_ip, PORT_META)
            if s is None:
                print_with_timestamp("DEBUG: get_connected_socket failed in Meta thread")
                continue
            while True:
                # send weights
                self.__to_launch_lock.acquire()  # WEIGHTS LOCK...........................................................
                if len(self.__to_launch) > 0:  # new experiments to send
                    if not wait_ack:
                        obj = self.__to_launch.pop(0)
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__to_launch_lock.release()
                            print_with_timestamp("DEBUG: select_and_send_or_close_socket failed in Meta")
                            break
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_META_TO_SERVER:
                            print_with_timestamp("INFO: ACK timed-out, breaking connection")
                            self.__to_launch_lock.release()
                            break
                self.__to_launch_lock.release()  # END LOCK.......................................................
                # checks for samples batch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp("DEBUG: poll failed in Meta thread")
                    break
                elif obj is not None and obj != 'ACK':  # received finished
                    print_with_timestamp(f"DEBUG INFO: Meta interface received obj")
                    recv_time = time.time()
                    self.__results_lock.acquire()  # LOCK.........................................................
                    self.__results += obj
                    self.__results_lock.release()  # END LOCK.....................................................
                    self.__must_launch_lock.acquire()
                    self.__must_launch = True
                    self.__must_launch_lock.release()
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_tiemout:
                    print_with_timestamp(f"DEBUG: Timeout in TrainerInterface, not received anything for too long")
                    break
                time.sleep(LOOP_SLEEP_TIME)
            s.close()

    def run(self):
        """
        Meta learner main loop
        """
        logger = LoggerWandbPareto(RUN_NAME)
        finished_experiments, pareto_front = load_network_files()
        launched_experiments = []

        if finished_experiments is None:
            print(f"DEBUG: no meta dataset found, starting new run")
            finished_experiments = []  # list of dictionaries
            pareto_front = []  # list of dictionaries, subset of finished_experiments
            meta_model = SurrogateModel()
            meta_model.to(META_MODEL_DEVICE)
        else:
            print(f"DEBUG: existing meta dataset loaded")
            print("training new surrogate model...")
            meta_model = SurrogateModel()
            meta_model.to(META_MODEL_DEVICE)
            meta_model.train()
            meta_model, meta_loss = train_surrogate(meta_model, deepcopy(finished_experiments))
            print(f"surrogate model loss: {meta_loss}")

        # main meta-learning procedure:
        prev_exp = {}

        while True:
            self.__must_launch_lock.acquire()
            if self.__must_launch:
                self.__must_launch = False
                self.__must_launch_lock.release()
                self.__results_lock.acquire()
                temp_results = deepcopy(self.__results)
                self.__results = []
                self.__results_lock.release()
                for res in temp_results:
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
                        finished_experiments[to_update]["software_cost"] = min(finished_experiments[to_update]["software_cost"], res["software_cost"])
                        pareto_front = update_pareto(finished_experiments[to_update]["software_cost"], pareto_front)
                    else:
                        pareto_front = update_pareto(res, pareto_front)
                        finished_experiments.append(res)
                    prev_exp = res

                num_experiment = len(finished_experiments) + len(launched_experiments)
                print("---")
                print(f"ITERATION N° {num_experiment}")

                exp = {}
                exps = []
                model_selected = False
                meta_model.eval()

                while not model_selected:
                    exp = {}

                    # sample model
                    config_dict, unrounded = sample_config_dict(name=RUN_NAME + "_" + str(num_experiment), previous_exp=prev_exp, all_exp=finished_experiments + launched_experiments)

                    nb_params = nb_parameters(config_dict)
                    if nb_params > MAX_NB_PARAMETERS:
                        continue

                    with torch.no_grad():
                        predicted_loss = meta_model(config_dict).item()

                    exp["cost_hardware"] = nb_params
                    exp["cost_software"] = predicted_loss
                    exp["config_dict"] = config_dict
                    exp["unrounded"] = unrounded

                    exps.append(exp)

                    if len(exps) >= NB_SAMPLED_MODELS_PER_ITERATION:
                        # select model
                        model_selected = True
                        exp = exp_max_pareto_efficiency(exps, pareto_front, finished_experiments)

                print(f"config: {exp['config_dict']}")
                print(f"nb parameters: {exp['cost_hardware']}")
                print(f"predicted loss: {exp['cost_software']}")

                self.__to_launch_lock.acquire()
                self.__to_launch.append(exp)
                self.__to_launch_lock.release()
                launched_experiments.append(exp)

                if len(finished_experiments) > 0 and prev_exp != {}:
                    print("training new surrogate model...")

                    meta_model = SurrogateModel()
                    meta_model.to(META_MODEL_DEVICE)

                    meta_model.train()
                    meta_model, meta_loss = train_surrogate(meta_model, deepcopy(finished_experiments))

                    print(f"surrogate model loss: {meta_loss}")

                    dump_network_files(finished_experiments, pareto_front)
                    logger.log(surrogate_loss=meta_loss, surprise=prev_exp["surprise"], all_experiments=finished_experiments, pareto_front=pareto_front)
            else:
                self.__must_launch_lock.release()
            time.sleep(LOOP_SLEEP_TIME)


# WORKER: ===================================


class Worker:
    def __init__(self, server_ip=None):

        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.recv_timeout = RECV_TIMEOUT_WORKER_FROM_SERVER

        self.__finished_exp = None
        self.__finished_exp_lock = Lock()
        self.__exp_to_run = None
        self.__exp_to_run_lock = Lock()

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"server IP: {self.server_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()
        self.run()

    def __run_thread(self):
        """
        Worker thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(SOCKET_TIMEOUT_CONNECT_WORKER, self.server_ip, PORT_WORKER)
            if s is None:
                print_with_timestamp("DEBUG: get_connected_socket failed in worker")
                continue
            while True:
                # send buffer
                self.__finished_exp_lock.acquire()  # BUFFER LOCK.............................................................
                if self.__finished_exp is not None:  # a new result is available
                    print_with_timestamp("DEBUG: new result available")
                    if not wait_ack:
                        obj = deepcopy(self.__finished_exp)
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__finished_exp_lock.release()
                            print_with_timestamp("DEBUG: select_and_send_or_close_socket failed in worker")
                            break
                        self.__finished_exp = None
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_WORKER_TO_SERVER:
                            print_with_timestamp("INFO: ACK timed-out, breaking connection")
                            self.__finished_exp_lock.release()
                            break
                self.__finished_exp_lock.release()  # END BUFFER LOCK.........................................................
                # checks for new experiments to launch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp(f"INFO: worker poll failed")
                    break
                elif obj is not None and obj != 'ACK':
                    print_with_timestamp(f"DEBUG INFO: worker received obj")
                    recv_time = time.time()
                    self.__exp_to_run_lock.acquire()  # LOCK.......................................................
                    self.__exp_to_run = obj
                    self.__exp_to_run_lock.release()  # END LOCK...................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_timeout:
                    print_with_timestamp(f"DEBUG: Timeout in worker, not received anything for too long")
                    break
                time.sleep(LOOP_SLEEP_TIME)
            s.close()

    def run(self):
        while True:
            self.__exp_to_run_lock.acquire()
            if self.__exp_to_run is not None:
                exp = deepcopy(self.__exp_to_run)
                self.__exp_to_run = None
                self.__exp_to_run_lock.release()

                predicted_loss = exp['cost_software']
                exp["cost_software"] = run(exp["config_dict"])
                exp['surprise'] = exp["cost_software"] - predicted_loss
                self.__finished_exp_lock.acquire()
                self.__finished_exp = deepcopy(exp)
                self.__finished_exp_lock.release()
            else:
                self.__exp_to_run_lock.release()
            time.sleep(LOOP_SLEEP_TIME)


def main(args):
    if args.server:
        print("INFO: now running: server")
        Server()
    elif args.worker:
        Worker(server_ip=IP_SERVER)
        print("INFO: now running: worker")
    elif args.meta:
        MetaLearner(server_ip=IP_SERVER)
        print("INFO: now running: meta")
    else:
        print("ERROR: wrong argument")
    while True:
        time.sleep(10.0)
        pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--meta', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    main(args)
