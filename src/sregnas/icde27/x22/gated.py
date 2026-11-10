
import argparse
import os
import subprocess
from pathlib import Path
from .common import read,write,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();receipt=read(a.receipt) if a.receipt.exists() else {}
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if receipt.get('status')!='OK':
        reason='qualification prerequisite unavailable or unsuccessful: '+str(a.receipt)


        if '--out' in command:
            out=Path(command[command.index('--out')+1]);out.mkdir(parents=True,exist_ok=False)
            write(out/'run.json',dict(status='skipped',at=now(),error=reason,
                prerequisite=str(a.receipt),prerequisite_status=receipt.get('status','not_run'),
                Q_attempt=0,timing_valid=False,dependent_command_started=False))
            return 0
        raise RuntimeError(reason)
    return subprocess.call(command,env=os.environ)

if __name__=='__main__':raise SystemExit(main())
