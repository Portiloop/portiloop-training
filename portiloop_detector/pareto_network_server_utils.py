import logging
import pickle
import socket
import time
from copy import deepcopy
from datetime import datetime
from threading import Lock, Thread

import select
from requests import get

IP_SERVER = "142.182.5.48"  # Yann = "45.74.221.204"; Nicolas = "142.182.5.48"
PORT_META = 6666
PORT_WORKER = 6667

WAIT_BEFORE_RECONNECTION = 60.0

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

LEN_QUEUE_TO_LAUNCH = 10

HEADER_SIZE = 12

BUFFER_SIZE = 4096

PRINT_BYTESIZES = True


def print_with_timestamp(s):
    x = datetime.now()
    sx = x.strftime("%x %X ")
    logging.debug(sx + str(s))


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


if __name__ == "__main__":
    Server()
    while True:
        time.sleep(10.0)
        pass
