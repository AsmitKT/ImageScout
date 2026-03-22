#Major\utils\devices.py
import torch
from contextlib import contextmanager


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"),"cuda"
    if hasattr(torch,"xpu"):
        try:
            if torch.xpu.is_available():
                dev_idx=torch.xpu.current_device() if hasattr(torch.xpu,"current_device") else 0
                return torch.device("xpu",dev_idx),"xpu"
        except Exception:
            pass
    return torch.device("cpu"),"cpu"


@contextmanager
def AMPContext(backend:str,enabled:bool):
    if not enabled:
        yield
        return
    if backend=="cuda":
        with torch.cuda.amp.autocast(True):
            yield
    elif backend=="xpu":
        with torch.autocast(device_type="xpu",dtype=torch.float16,enabled=True):
            yield
    else:
        yield


class _NullScaler:
    def is_enabled(self): return False
    def scale(self,loss): return loss
    def step(self,opt): opt.step()
    def update(self): pass


def get_grad_scaler(backend:str,enabled:bool):
    if not enabled:
        return _NullScaler()
    if backend=="cuda":
        return torch.cuda.amp.GradScaler(enabled=True)
    if backend=="xpu":
        return _NullScaler()
    return _NullScaler()