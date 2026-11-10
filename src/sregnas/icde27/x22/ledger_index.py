
import json
import os

_cache={}

def previous_identities(path,stream):
    stat=os.fstat(stream.fileno());key=str(path)
    cached=_cache.get(key)
    if cached is None or cached[0]!=(stat.st_dev,stat.st_ino) or cached[1]>stat.st_size:
        cached=[(stat.st_dev,stat.st_ino),0,set()];_cache[key]=cached
    stream.seek(cached[1])
    for line in stream:
        if line.strip():
            row=json.loads(line);cached[2].add((row['identity'],row['stage']))
    cached[1]=stream.tell()
    return cached[2]
