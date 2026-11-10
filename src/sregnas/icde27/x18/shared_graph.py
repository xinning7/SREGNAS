
import json
from pathlib import Path
import tempfile

SCHEMA='x18_shared_payload_v2'


class Store:
    def __init__(self,owner,root=None):
        self.root=Path(tempfile.mkdtemp(prefix='x18_v2_',dir=str(root or '/dev/shm')))
        self.number=0
        (self.root/'owner.json').write_text(json.dumps(dict(experiment='X18',owner=str(owner),version='x18_v2')))

    def array(self,value):
        import numpy as np
        import torch
        a=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
        path=self.root/f'a{self.number:05d}.npy';self.number+=1
        shared=np.lib.format.open_memmap(path,mode='w+',dtype=a.dtype,shape=a.shape)
        shared[...]=a
        return dict(_x18_array=str(path),shape=list(a.shape),dtype=str(a.dtype))

    def tree(self,value):
        import numpy as np
        import torch
        if isinstance(value,(np.ndarray,torch.Tensor)):return self.array(value)
        if isinstance(value,dict):return {k:self.tree(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [self.tree(v) for v in value]
        if isinstance(value,np.generic):return value.item()
        return value

    def payload(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(schema=SCHEMA,shared_root=str(self.root),payload=self.tree(value))))
        return str(path)


def load_payload(path):
    import torch
    path=Path(path)
    if not path.name.endswith('.x18-shared.json'):
        return torch.load(path,map_location='cpu',weights_only=False)
    record=json.loads(path.read_text())
    if record.get('schema')!=SCHEMA:raise ValueError('invalid X18 shared payload descriptor')
    def unpack(value):
        import numpy as np
        if isinstance(value,dict) and '_x18_array' in value:
            array=np.load(value['_x18_array'],mmap_mode='c')
            if list(array.shape)!=value['shape'] or str(array.dtype)!=value['dtype']:
                raise ValueError('shared array metadata mismatch')
            return torch.from_numpy(array)
        if isinstance(value,dict):return {k:unpack(v) for k,v in value.items()}
        if isinstance(value,list):return [unpack(v) for v in value]
        return value
    return unpack(record['payload'])
