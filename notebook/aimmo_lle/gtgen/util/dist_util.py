import torch
import torch.distributed as dist

class DistributedStop:
    def __init__(self, distributed, rank, log) -> None:
        self.distributed = distributed
        self.rank = rank
        self.logger = log

        self.stop_signal = 0

    def set_stop(self):
        self.stop_signal = 1

    def reduce_stop_event(self):
        if self.distributed:
            # dist.all_reduce(self.stop_event, op=dist.ReduceOp.SUM, group=self.group)
            stop_signal_t = torch.tensor(self.stop_signal).cuda()
            dist.all_reduce(stop_signal_t, op=dist.ReduceOp.SUM)
            check = int(stop_signal_t.cpu().detach().numpy())
        else:
            check = self.stop_signal

        if check>0:
            self.logger.info(f"reduce_stop_event. rank={self.rank} process. stop_event={check}")
        return check>0
