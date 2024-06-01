from abc import ABC, abstractmethod
import logging
import os
import subprocess
import sys
import signal

from polymetis.utils.grpc_utils import check_server_exists
from polymetis.utils.data_dir import BUILD_DIR, which

class BaseServerInterface(ABC):
    def __init__(self, 
                 name: str,
                 ip: str, 
                 port: int):
        self.name = name
        self.ip = ip
        self.port = port

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
                 use_real_time: bool = True,
                 robot_model="franka_panda",
                 robot_client="franka_hardware",
                 server_exec="run_server"):
        name = f"{robot_name}_server"
        super().__init__(name, ip, port)
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
        self.logger.info(f"Starting server...")
        server_exec_path = which(self.server_exec)
        server_cmd = [server_exec_path]
        server_cmd = server_cmd + ["-s", self.ip, "-p", self.port]

        if self.use_real_time:
            server_cmd += ["-r"]

        server_output = subprocess.Popen(
            server_cmd, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setpgrp
        )
        self.pgid = os.getpgid(server_output.pid)

        return server_output

    def stop(self):
        if self.pgid is not None:
            self.logger.info(f"Killing subprocess with pid {self.server_output.pid}, pgid {self.pgid}...")
            os.killpg(self.pgid, signal.SIGINT)