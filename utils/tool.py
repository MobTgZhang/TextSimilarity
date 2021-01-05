import json
import time
import os
import logging
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
class DataSaver(object):
    """save every epoch datas."""
    def __init__(self,model_name,mode):
        self.loss_list = []
        self.f1_list = []
        self.em_list = []
        self.model_name = model_name + "_" + mode
        self.mode = mode
    def add_f1(self,val):
        val = float(val)
        self.f1_list.append(val)
    def add_loss(self,val):
        val = float(val)
        self.loss_list.append(val)
    def add_em(self,val):
        val = float(val)
        self.em_list.append(val)
    def save(self,save_path):
        save_file = os.path.join(save_path, self.model_name + "_data.json")
        data = {"f1_score":self.f1_list,
                "em_score":self.em_list,
                "loss":self.loss_list}
        with open(save_file,mode="w",encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False)
        logger.info("saved file: %s"% save_file)