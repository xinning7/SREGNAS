
from .common import OUT
from sregnas.icde27.x18 import quality_worker as worker

worker.OUT=OUT
worker.VERSION='x25'
_write=worker.write
def write(path,value):
    if isinstance(value,dict) and str(value.get('run_id','')).startswith('X18_'):
        value.update(run_id='expX25_'+value['run_id'][4:],experiment_id='X25')
    return _write(path,value)
worker.write=write

if __name__=='__main__':worker.main()
