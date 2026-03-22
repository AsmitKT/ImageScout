#Major\utils\samplers.py
import random
from torch.utils.data import Sampler


class GroupByFilenameBatchSampler(Sampler):
    def __init__(self,df,batch_size,drop_last=False,seed=42):
        assert batch_size%5==0,"Use batch_size multiple of 5"
        self.batch_size=batch_size
        self.drop_last=drop_last
        self.seed=seed
        groups={}
        for idx,fn in enumerate(df["filename"].tolist()):
            groups.setdefault(fn,[]).append(idx)
        self.groups=[idxs for _,idxs in groups.items()]
        self.epoch=0
    def set_epoch(self,e):
        self.epoch=int(e)
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        order=list(range(len(self.groups)))
        rng.shuffle(order)
        batch=[]
        for gi in order:
            batch.extend(self.groups[gi])
            while len(batch)>=self.batch_size:
                yield batch[:self.batch_size]
                batch=batch[self.batch_size:]
        if len(batch) and not self.drop_last:
            yield batch
    def __len__(self):
        from math import ceil
        total=sum(len(g) for g in self.groups)
        return ceil(total/self.batch_size)


class UniqueFilenameBatchSampler(Sampler):
    def __init__(self,df,batch_size,drop_last=False,seed=42):
        self.batch_size=batch_size
        self.drop_last=drop_last
        self.seed=seed
        fn_to_indices={}
        for idx,fn in enumerate(df["filename"].tolist()):
            fn_to_indices.setdefault(fn,[]).append(idx)
        self.fn_to_indices=fn_to_indices
        self.filenames=list(fn_to_indices.keys())
        self.epoch=0
    def set_epoch(self,e):
        self.epoch=int(e)
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        order=list(self.filenames)
        rng.shuffle(order)
        batch=[]
        for fn in order:
            idxs=self.fn_to_indices[fn]
            j=rng.randrange(len(idxs))
            batch.append(idxs[j])
            if len(batch)==self.batch_size:
                yield batch
                batch=[]
        if len(batch) and not self.drop_last:
            yield batch
    def __len__(self):
        from math import ceil
        return ceil(len(self.filenames)/self.batch_size)