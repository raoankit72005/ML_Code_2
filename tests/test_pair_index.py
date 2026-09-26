"""SQLite regression: indexed join, preserved batch ordering, safe step-zero repair."""
import ast
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('repair_pair_index',ROOT/'scripts/repair_pair_index.py')
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)

class PairIndexTests(unittest.TestCase):
    def fixture(self,root):
        conn=sqlite3.connect(root/'clusters.sqlite')
        conn.executescript('CREATE TABLE queries(sid TEXT PRIMARY KEY,cluster TEXT,split TEXT,country TEXT,ids TEXT,shuffle TEXT); CREATE TABLE pairs(sid TEXT,eid TEXT,ordinal INTEGER); CREATE INDEX pair_order ON pairs(ordinal,sid); CREATE INDEX query_split ON queries(split,country,shuffle);')
        conn.executemany('INSERT INTO queries VALUES(?,?,?,?,?,?)',((f'q{i}',f'c{i}', 'train','IN' if i%2 else 'US','[]',str(1000-i)) for i in range(1000)))
        conn.executemany('INSERT INTO pairs VALUES(?,?,?)',((f'q{i}',f'e{i}-{j}',j) for i in range(1000) for j in range(3)))
        conn.commit();return conn

    def test_repair_preserves_every_ordered_pair(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root)
            before=c.execute(repair.SQL,('train',)).fetchall();c.close()
            repair.repair(root)
            c=sqlite3.connect(root/'clusters.sqlite')
            self.assertEqual(before,c.execute(repair.SQL,('train',)).fetchall())
            plan=list(c.execute('EXPLAIN QUERY PLAN '+repair.SQL,('train',)))
            self.assertTrue(any('SEARCH p' in r[3] and 'pair_sid' in r[3] for r in plan),plan)
            c.close();repair.repair(root)  # Repeatable, without modifying rows.

    def test_existing_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root).close()
            (root/'encoder').mkdir();(root/'encoder/last.pt').touch()
            with self.assertRaisesRegex(RuntimeError,'checkpoint'):repair.repair(root)

    def test_builder_creates_sid_index(self):
        tree=ast.parse((ROOT/'src/er_pipeline/clusters.py').read_text())
        scripts=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='executescript' and isinstance(n.args[0],ast.Constant)]
        c=sqlite3.connect(':memory:')
        for sql in scripts:c.executescript(sql)
        self.assertEqual([r[2] for r in c.execute('PRAGMA index_info(pair_sid)')],['sid','ordinal','eid'])
        c.close()
