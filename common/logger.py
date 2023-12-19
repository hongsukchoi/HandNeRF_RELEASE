import logging
import os

OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'

PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING

class colorlogger():
    def __init__(self, log_dir, log_name='train_logs.txt'):
        # set log
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.INFO)
        log_file = os.path.join(log_dir, log_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_log = logging.FileHandler(log_file, mode='a')
        file_log.setLevel(logging.INFO)
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "{}%(asctime)s{} %(message)s".format(GREEN, END),
            "%m-%d %H:%M:%S")
        file_log.setFormatter(formatter)
        console_log.setFormatter(formatter)
        self._logger.addHandler(file_log)
        self._logger.addHandler(console_log)

    def debug(self, msg):
        self._logger.debug(str(msg))

    def info(self, msg):
        self._logger.info(str(msg))

    def warning(self, msg):
        self._logger.warning(WARNING + 'WRN: ' + str(msg) + END)

    def critical(self, msg):
        self._logger.critical(RED + 'CRI: ' + str(msg) + END)

    def error(self, msg):
        self._logger.error(RED + 'ERR: ' + str(msg) + END)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.history = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, epoch=0):

        # make sure the history is of the same len as epoch
        while len(self.history) <= epoch:
            self.history.append([])

        self.history[epoch].append(val / n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_epoch_averages(self, epoch=-1):
        if len(self.history) == 0:  # no stats here
            return None
        elif epoch == -1:
            return [
                (float(np.array(x).mean()) if len(x) > 0 else float("NaN"))
                for x in self.history
            ]
        else:
            return float(np.array(self.history[epoch]).mean())

    def get_all_values(self):
        all_vals = [np.array(x) for x in self.history]
        all_vals = np.concatenate(all_vals)
        return all_vals

    def get_epoch(self):
        return len(self.history)

    @staticmethod
    def from_json_str(json_str):
        self = AverageMeter()
        self.__dict__.update(json.loads(json_str))
        return self
