#Major\utils\schedule.py
import math
class TwoPhaseCosineEpochsScaled:
    def __init__(self,p1_epochs=150,p1_warmup=5,p2_epochs=12,p2_warmup=2,min_lr_mult=0.0,p2_lr_scale=0.8):
        self.p1=float(max(1,p1_epochs));self.w1=float(max(0,p1_warmup))
        self.p2=float(max(1,p2_epochs));self.w2=float(max(0,p2_warmup))
        self.min=float(min_lr_mult);self.s2=float(p2_lr_scale)
    def _seg(self,e,tot,warm):
        if e<warm:
            return e/max(1.0,warm)
        span=max(1.0,tot-warm)
        x=min(max(0.0,e-warm),span)
        return self.min+(1.0-self.min)*0.5*(1.0+math.cos(math.pi*x/span))
    def lr_mult(self,epoch_idx,batch_idx,steps_per_epoch):
        e=float(epoch_idx)+float(batch_idx)/max(1,steps_per_epoch)
        if e<self.p1:
            return self._seg(e,self.p1,self.w1)
        return self.s2*self._seg(e-self.p1,self.p2,self.w2)