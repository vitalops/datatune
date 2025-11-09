import logging
import threading
import time
import sys


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",   # white
        "INFO": "\033[36m",    # cyan
        "WARNING": "\033[33m", # yellow
        "ERROR": "\033[31m",   # red
        "CRITICAL": "\033[41m",# red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def get_logger(name: str = None):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = ColorFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
class Spinner:
    def __init__(self, message="Processing..."):
        self.message = message
        self.spinner = ['|', '/', '-', '\\']
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def _spin(self):
        i = 0
        while self.running:
            sys.stdout.write(f"\r{self.message} {self.spinner[i % len(self.spinner)]}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()