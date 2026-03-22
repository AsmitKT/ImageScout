#Major\utils\eval_logger.py
import os,json,datetime
class EvalLogger:
    def __init__(self,ckpt_dir,filename="eval_metrics.jsonl"):
        self.path=os.path.join(ckpt_dir,filename)
        os.makedirs(ckpt_dir,exist_ok=True)
    def write(self,epoch,metrics,extras=None):
        row={"ts":datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z","epoch":int(epoch)}
        if isinstance(metrics,dict): row.update(metrics)
        if isinstance(extras,dict): row.update(extras)
        with open(self.path,"a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    def pathstr(self):
        return self.path