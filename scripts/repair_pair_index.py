"""Repair the step-0 batching bottleneck without changing code or data rows.
Stop the pipeline first. This repair refuses trained encoder checkpoints because
changing the SQLite file invalidates their recorded input identity.
"""
import argparse
import os
from pathlib import Path
import shutil
import sqlite3
import time

SQL = '''SELECT p.sid,p.eid,q.cluster,q.country
FROM pairs p JOIN queries q USING(sid) WHERE q.split=?
ORDER BY q.country,p.ordinal,q.shuffle,p.sid,p.eid'''

def repair(work):
    work=work.resolve()
    db=work/'clusters.sqlite'
    if not db.is_file():raise RuntimeError(f'Missing database: {db}')
    for name in ('last.pt','complete.json'):
        if (work/'encoder'/name).exists():
            raise RuntimeError(f'encoder/{name} exists. This repair is only for the initial step-0 stall; preserve the checkpoint and request a checkpoint-aware migration.')
    # Refuse the known running pipeline before opening a write transaction.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(Path(os.fsdecode(a)).name in ('hybrid.py','gpu_worker.py','run.py') for a in args if a):
            if os.fsencode(str(work)) in args:
                raise RuntimeError(f'Pipeline PID {entry.name} is still using this work directory. Stop it first.')
    free=shutil.disk_usage(work).free
    required=4*db.stat().st_size+2*1024**3
    if free<required:
        raise RuntimeError(f'Need conservative index-build headroom {required/1024**3:.1f} GiB; only {free/1024**3:.1f} GiB free')
    temp=work/'repair_tmp';temp.mkdir(exist_ok=True)
    os.environ['SQLITE_TMPDIR']=str(temp)
    start=time.monotonic();last=[start]
    conn=sqlite3.connect(db.as_uri()+'?mode=rw',uri=True,timeout=5)
    try:
        conn.execute('PRAGMA temp_store=FILE')
        conn.execute('PRAGMA cache_size=-65536')
        print('Before:',list(conn.execute('EXPLAIN QUERY PLAN '+SQL,('train',))),flush=True)
        def progress():
            now=time.monotonic()
            if now-last[0]>=10:
                print(f'Building join index: {now-start:.0f}s elapsed',flush=True);last[0]=now
            return 0
        conn.set_progress_handler(progress,100000)
        conn.execute('BEGIN EXCLUSIVE')
        conn.execute('CREATE INDEX IF NOT EXISTS pair_sid ON pairs(sid,ordinal,eid)')
        columns=[r[2] for r in conn.execute('PRAGMA index_info(pair_sid)')]
        if columns!=['sid','ordinal','eid']:raise RuntimeError('Existing pair_sid index has unexpected columns')
        conn.commit()
        plan=list(conn.execute('EXPLAIN QUERY PLAN '+SQL,('train',)))
        print('After:',plan,flush=True)
        if not any('SEARCH p' in r[3] and 'pair_sid' in r[3] for r in plan):
            raise RuntimeError('Index exists, but query planner did not choose it. Send the printed plan; do not restart yet.')
        print('REPAIR OK. Data rows and source code were not changed. Restart the same command with the same work directory.',flush=True)
    except BaseException:
        conn.rollback();raise
    finally:conn.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    repair(parser.parse_args().work)
