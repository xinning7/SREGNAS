
import atexit
import os
from pathlib import Path
import sqlite3
import threading

_stores={}
_registry_lock=threading.RLock()

class Store:
    def __init__(self,path):
        self.path=Path(path);self.lock=threading.RLock();self.references=0
        self.conn=sqlite3.connect(':memory:',isolation_level=None,check_same_thread=False)
        self.conn.row_factory=sqlite3.Row
        if self.path.exists():
            with sqlite3.connect(self.path) as source:source.backup(self.conn)
        self.dirty=False
    def flush(self):
        with self.lock:
            if not self.dirty:return
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.path.with_name(self.path.name+f'.x22-{os.getpid()}.tmp')
            with sqlite3.connect(temporary) as dest:self.conn.backup(dest)
            temporary.replace(self.path);self.dirty=False

class Cursor:
    def __init__(self,rows,rowcount,lastrowid,description):
        self._rows=iter(rows);self.rowcount=rowcount;self.lastrowid=lastrowid;self.description=description
    def fetchone(self):return next(self._rows,None)
    def fetchall(self):return list(self._rows)
    def __iter__(self):return self._rows

class Connection:
    def __init__(self,store):self.store=store;self.closed=False;store.references+=1
    @property
    def row_factory(self):return self.store.conn.row_factory
    @row_factory.setter
    def row_factory(self,value):self.store.conn.row_factory=value
    def _call(self,method,*args,**kwargs):
        if self.closed:raise sqlite3.ProgrammingError('closed X22 queue connection')
        with self.store.lock:
            cursor=getattr(self.store.conn,method)(*args,**kwargs)

            rows=cursor.fetchall() if cursor.description else []
            self.store.dirty=True
            return Cursor(rows,cursor.rowcount,cursor.lastrowid,cursor.description)
    def execute(self,*args,**kwargs):return self._call('execute',*args,**kwargs)
    def executemany(self,*args,**kwargs):return self._call('executemany',*args,**kwargs)
    def executescript(self,*args,**kwargs):return self._call('executescript',*args,**kwargs)
    def commit(self):pass
    def close(self):
        with _registry_lock:
            if self.closed:return
            self.closed=True;self.store.references-=1
            if self.store.references==0:self.store.flush()

def connect(path):
    key=str(Path(path).resolve())
    with _registry_lock:
        if key not in _stores:_stores[key]=Store(key)
        return Connection(_stores[key])

def flush():
    for store in list(_stores.values()):store.flush()

atexit.register(flush)
