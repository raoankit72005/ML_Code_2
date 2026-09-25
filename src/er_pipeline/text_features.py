"""Deterministic hashed TF-IDF fitted only to train S2/S3; no dense pair matrix."""
import hashlib
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np


def words(s):
    return s.split()


def grams(s, n=3):
    # Word-boundary grams are insensitive to whole-word ordering.
    return {w[i:i+n] for token in words(s) for w in [' ' + token + ' ']
            for i in range(max(1, len(w)-n+1))}


def terms(s):
    for word in words(s):
        yield 'w:' + word
        padded = ' ' + word + ' '
        for n in (3, 4, 5):
            for i in range(max(0, len(padded)-n+1)):
                yield f'{n}:' + padded[i:i+n]


@lru_cache(maxsize=65536)
def bucket(term, bits):
    return int.from_bytes(hashlib.blake2b(term.encode('utf-8'), digest_size=8).digest(), 'little') & ((1 << bits)-1)


def name_text(r):
    return r['name_latin_folded'] or r['name_without_suffix'] or r['name_basic']


def address_text(r):
    return r['address_latin_folded'] or r['address_normalized']


class Tfidf:
    def __init__(self, bits=18):
        self.bits = bits
        self.n = 0
        self.df = {k: np.zeros(1 << bits, dtype=np.uint64) for k in ('name', 'address')}
        self.idf = None
    def update(self, record):
        self.n += 1
        for field, text in [('name', name_text(record)), ('address', address_text(record))]:
            indices = list({bucket(t, self.bits) for t in terms(text)})
            self.df[field][indices] += 1
    def finish(self):
        self.idf = {k: (np.log((self.n + 1) / (v.astype(np.float64) + 1)) + 1).astype(np.float32) for k, v in self.df.items()}
    def save(self, path):
        self.finish()
        np.savez_compressed(path, bits=self.bits, n=self.n, name=self.idf['name'], address=self.idf['address'])
    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            obj = cls(int(d['bits']))
            obj.n = int(d['n'])
            obj.idf = {k: d[k].copy() for k in ('name', 'address')}
        obj.df = None
        return obj
    def vector(self, text, field):
        counts = Counter(bucket(t, self.bits) for t in terms(text))
        values = {i: (1 + math.log(n)) * float(self.idf[field][i]) for i, n in counts.items()}
        norm = math.sqrt(sum(v*v for v in values.values()))
        return {i: v / norm for i, v in values.items()} if norm else {}
    def query_tokens(self, text, field, limit):
        # Retrieval uses WORDS + 3-grams, ranked by frozen training IDF.
        options = {'w:' + w for w in words(text)} | {'3:' + g for g in grams(text)}
        ordered = sorted(options, key=lambda t: (-float(self.idf[field][bucket(t, self.bits)]), t))
        return [encoded(t) for t in ordered[:limit]]


def cosine(a, b):
    if not a or not b:
        return math.nan
    if len(a) > len(b):
        a, b = b, a
    return min(1.0, max(0.0, sum(v * b.get(i, 0) for i, v in a.items())))


def encoded(s):
    # Exact UTF-8 hex encoding: no retrieval-token hash collisions.
    return 'x' + s.encode('utf-8').hex()


def lexical_tokens(text):
    return ' '.join(sorted({encoded('w:' + w) for w in words(text)} | {encoded('3:' + g) for g in grams(text)}))


def block_keys(r):
    name = name_text(r)
    prefix = ''.join(name.split())[:3]
    city, state, postal = r['city_candidate'], r['state_candidate'], r['postal_code_candidate']
    house = r['house_number_zero_normalized']
    keys = {'A': [], 'B': [], 'C': [], 'D': []}
    def add(channel, *values):
        if all(values):
            keys[channel].append(encoded(channel + '\x1f' + '\x1f'.join(values)))
    add('A', 'basic', r['name_basic'])
    add('A', 'core', name)
    add('A', 'suffix_free', r['name_without_suffix'])
    add('A', 'expanded', r['name_expanded'])
    add('A', 'sorted', ' '.join(sorted(name.split())))
    if len(prefix) == 3:
        add('B', prefix)
        add('D', house, prefix)
    add('C', 'postal', state, postal)
    # Country restriction is applied by retrieval; state extraction can be absent.
    add('C', 'postal_only', postal)
    if city:
        stop = {'road','street','avenue','lane','block','near','and','the'}
        for token in sorted(set(address_text(r).split()) - stop):
            if len(token) >= 3 and not token.isdigit() and token not in city.split():
                add('C', 'city_token', city, token)
        add('C', 'city_name', city, ' '.join(name.split()[:3]))
    return keys
