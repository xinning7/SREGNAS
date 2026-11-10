
import csv
import subprocess


def inventory():
    def query(args):
        return list(csv.reader(subprocess.check_output(['nvidia-smi',*args],text=True).splitlines()))
    cards=[dict(gpu=int(r[0]),uuid=r[1].strip(),name=r[2].strip(),total_gib=float(r[3])/1024,
                free_gib=float(r[4])/1024,compute_mode=r[5].strip(),processes=[])
           for r in query(['--query-gpu=index,uuid,name,memory.total,memory.free,compute_mode','--format=csv,noheader,nounits'])]
    by_uuid={c['uuid']:c for c in cards}
    for r in query(['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']):
        if len(r)==4 and r[0].strip() in by_uuid:
            by_uuid[r[0].strip()]['processes'].append(dict(pid=int(r[1]),name=r[2].strip(),used_mib=r[3].strip()))
    return cards


def eligible(card, peak, *, empty=False):
    if 'RTX A6000' not in card['name']: return False
    if card['compute_mode']!='Default' and card['processes']: return False
    if empty or peak+4>card['total_gib']: return not card['processes']
    return card['free_gib']>=peak+4
