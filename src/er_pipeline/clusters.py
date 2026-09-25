"""Disk-backed ground-truth connected components and deterministic cluster splits."""
import hashlib
import json
from .common import connect, rows, truth_ids, lookup


def build(index_path, truth_path, destination, fraction=.2, seed=42):
    if destination.exists(): destination.unlink()
    conn=connect(destination)
    conn.executescript('''CREATE TABLE nodes(sid TEXT PRIMARY KEY,parent TEXT,size INTEGER);
    CREATE TABLE owners(eid TEXT PRIMARY KEY,sid TEXT);
    CREATE TABLE queries(sid TEXT PRIMARY KEY,cluster TEXT,split TEXT,country TEXT,ids TEXT,shuffle TEXT);
    CREATE TABLE pairs(sid TEXT,eid TEXT,ordinal INTEGER);
    ''')
    refs=connect(index_path,readonly=True)
    def root(sid):
        trail=[]
        while True:
            parent=conn.execute('SELECT parent FROM nodes WHERE sid=?',(sid,)).fetchone()[0]
            if parent==sid: break
            trail.append(sid);sid=parent
        for child in trail:conn.execute('UPDATE nodes SET parent=? WHERE sid=?',(sid,child))
        return sid
    def union(a,b):
        a,b=root(a),root(b)
        if a==b:return
        sa=conn.execute('SELECT size FROM nodes WHERE sid=?',(a,)).fetchone()[0]
        sb=conn.execute('SELECT size FROM nodes WHERE sid=?',(b,)).fetchone()[0]
        if sa<sb:a,b,sa,sb=b,a,sb,sa
        conn.execute('UPDATE nodes SET parent=? WHERE sid=?',(a,b))
        conn.execute('UPDATE nodes SET size=? WHERE sid=?',(sa+sb,a))
    for i,q in enumerate(rows(truth_path),1):
        sid=q['source1_entity_id'];ids=truth_ids(q['matched_entity_ids'])
        country=lookup(refs,sid)['country_key']
        conn.execute('INSERT INTO nodes VALUES(?,?,1)',(sid,sid))
        conn.execute('INSERT INTO queries VALUES(?,NULL,NULL,?,?,NULL)',(sid,country,json.dumps(ids)))
        for j,eid in enumerate(ids):
            if not refs.execute('SELECT 1 FROM records WHERE entity_id=? AND source IN (2,3)',(eid,)).fetchone():
                raise ValueError(f'Missing truth reference {eid}')
            conn.execute('INSERT INTO pairs VALUES(?,?,?)',(sid,eid,j))
            owner=conn.execute('SELECT sid FROM owners WHERE eid=?',(eid,)).fetchone()
            if owner:union(sid,owner[0])
            else:conn.execute('INSERT INTO owners VALUES(?,?)',(eid,sid))
        if i%10000==0:conn.commit();print(f'Cluster construction: {i:,} queries',flush=True)
    # Only final roots determine split; shared references cannot bridge partitions.
    for (sid,) in conn.execute('SELECT sid FROM nodes ORDER BY sid'):
        cluster=root(sid)
        digest=hashlib.blake2b(f'{seed}|{cluster}'.encode(),digest_size=8).hexdigest()
        split='validation' if int(digest,16)/2**64<fraction else 'train'
        conn.execute('UPDATE queries SET cluster=?,split=?,shuffle=? WHERE sid=?',(cluster,split,digest,sid))
    conn.executescript('CREATE INDEX pair_order ON pairs(ordinal,sid); CREATE INDEX query_split ON queries(split,country,shuffle);')
    conn.commit();refs.close();conn.close()
