from abc import ABC, abstractmethod
import logging
import os
import subprocess
import signal
import time
import hydra

from polymetis.utils.grpc_utils import check_server_exists
from polymetis.utils.data_dir import BUILD_DIR, which

class BaseServerInterface(ABC):
    def __init__(self, 
                 name: str,
                 ip: str, 
                 port: int,
                 timeout: int = 15):
        self.name = name
        self.ip = ip
        self.port = port
        self.timeout = timeout

        self.logger = logging.getLogger(self.name)
        self.pgid = None

    def __del__(self):
        self.stop()

    @abstractmethod 
    def start(self) -> subprocess.Popen:
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

class RobotServerInterface(BaseServerInterface):
    def __init__(self, 
                 ip:str, 
                 port:int, 
                 robot_name:str,
                 timeout:int = 15,
                 use_real_time: bool = True,
                 robot_model="franka_panda",
                 robot_client="franka_hardware",
                 server_exec="run_server"):
        name = f"{robot_name}_server"
        super().__init__(name, ip, port, timeout=timeout)
        self.use_real_time = use_real_time
        self.robot_model = robot_model
        self.robot_client = robot_client
        self.server_exec = server_exec

        self.logger.info(f"Adding {BUILD_DIR} to $PATH")
        os.environ["PATH"] = BUILD_DIR + os.pathsep + os.environ["PATH"]

    def start(self) -> subprocess.Popen:
        # Check if another server is alive on address
        assert not check_server_exists(
            self.ip, self.port
        ), (
            "Port unavailable; possibly another server found on designated address. "
            "To prevent undefined behavior, start the service on a different port or "
            "kill stale servers with 'pkill -9 run_server'"
        )

        # start server
        self.logger.info(f"Starting server...")
        server_exec_path = which(self.server_exec)
        server_cmd = [server_exec_path]
        server_cmd = server_cmd + ["-s", self.ip, "-p", str(self.port)]

        if self.use_real_time:
            server_cmd += ["-r"]

        server_output = subprocess.Popen(
            server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )
        self.pgid = os.getpgid(server_output.pid)

        # start client
        t0 = time.time()
        while not check_server_exists(self.ip, self.port):
            time.sleep(0.1)
            if time.time() - t0 > self.timeout:
                raise ConnectionError("Robot client: Unable to locate server.")
        self.logger.info("Starting robot client...")
        client = hydra.utils.instantiate(self.robot_client)
        client.run()


        return server_output

    def stop(self):
        if self.pgid is not None:
            self.logger.info(f"Killing subprocess with pid {self.server_output.pid}, pgid {self.pgid}...")
            os.killpg(self.pgid, signal.SIGINT)

    @staticmethod
    def stop_all_servers():
        """stops all running robot servers.
        """
        command = ["pkill", "-9", "run_server"]
        subprocess.run(command)