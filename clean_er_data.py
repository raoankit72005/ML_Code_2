#!/usr/bin/env python3
"""Streaming business entity-resolution cleaning. Python 3.9+, standard library only.

Run: python clean_er_data.py --input ML_Dataset.zip --output cleaned_data
Also accepts an extracted dataset folder. Outputs gzip TSVs and a JSON report.
Original fields, IDs, ordering and duplicate-looking records are preserved.
Ground truth is copied unchanged. No external lookups or learned transformations.

Address components are heuristic candidates, not verified addresses. Do not use
number disagreement as a hard rejection. Country is a candidate-generation key;
unknown countries need fallback handling. No transliteration is attempted.
Import pair_features() to compute comparison features AFTER generating pairs.
Cleaning alone does not produce predictions or guarantee a leaderboard score.
"""
import argparse
import csv
import gzip
import html
import io
import json
import math
import re
import shutil
import unicodedata as ud
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

MISSING = {'', 'null', '<null>', 'none', '<none>', 'n/a', 'na', 'nan', '<na>', '-', '--'}
LEGAL = {'pvt': 'private', 'ltd': 'limited', 'corp': 'corporation',
         'co': 'company', 'inc': 'incorporated', 'llp': 'llp', 'llc': 'llc'}
SUFFIXES = set(LEGAL.values()) | {'plc', 'sarl', 'sas', 'sasu', 'eurl'}
COUNTRIES = {'india': 'IN', 'in': 'IN', 'ind': 'IN', 'us': 'US', 'usa': 'US',
             'united states': 'US', 'united states of america': 'US',
             'fr': 'FR', 'fra': 'FR', 'france': 'FR'}
US_STATES = dict(pair.split(':') for pair in (
 'AL:alabama|AK:alaska|AZ:arizona|AR:arkansas|CA:california|CO:colorado|'
 'CT:connecticut|DE:delaware|FL:florida|GA:georgia|HI:hawaii|ID:idaho|'
 'IL:illinois|IN:indiana|IA:iowa|KS:kansas|KY:kentucky|LA:louisiana|'
 'ME:maine|MD:maryland|MA:massachusetts|MI:michigan|MN:minnesota|'
 'MS:mississippi|MO:missouri|MT:montana|NE:nebraska|NV:nevada|'
 'NH:new hampshire|NJ:new jersey|NM:new mexico|NY:new york|NC:north carolina|'
 'ND:north dakota|OH:ohio|OK:oklahoma|OR:oregon|PA:pennsylvania|'
 'RI:rhode island|SC:south carolina|SD:south dakota|TN:tennessee|TX:texas|'
 'UT:utah|VT:vermont|VA:virginia|WA:washington|WV:west virginia|'
 'WI:wisconsin|WY:wyoming|DC:district of columbia').split('|'))
INDIA_STATES = ('andhra pradesh|arunachal pradesh|assam|bihar|chhattisgarh|goa|'
 'gujarat|haryana|himachal pradesh|jharkhand|karnataka|kerala|madhya pradesh|'
 'maharashtra|manipur|meghalaya|mizoram|nagaland|odisha|punjab|rajasthan|'
 'sikkim|tamil nadu|telangana|tripura|uttar pradesh|uttarakhand|west bengal|'
 'delhi|chandigarh|puducherry|jammu and kashmir|ladakh|lakshadweep|'
 'andaman and nicobar islands|dadra and nagar haveli and daman and diu').split('|')
ROAD = {'rd': 'road', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard',
        'ln': 'lane', 'hwy': 'highway'}
DOTTED = re.compile(r'(?<!\w)(?:[a-z]\.){2,}(?:[a-z](?!\w))?')
ID_TAG = re.compile(r'\(\s*id\s*:\s*[^)]*\)|\[\s*id\s*:\s*[^]]*\]', re.I)


def unicode_normalize(value):
    s = ud.normalize('NFKC', html.unescape(value)).casefold()
    # Preserve Indic combining marks and join controls used by some scripts.
    s = ''.join(' ' if ud.category(c) == 'Cc' else c for c in s)
    s = ''.join(c for c in s if ud.category(c) != 'Cf' or c in '\u200c\u200d')
    return ' '.join(s.split())


@lru_cache(maxsize=4096)
def keep_character(c):
    return ud.category(c)[0] in 'LMN' or c in '\u200c\u200d'


def basic(value, address=False):
    s = unicode_normalize(value)
    s = DOTTED.sub(lambda m: m[0].replace('.', ''), s)
    s = s.replace('&', ' and ')
    # Only retain slash/hyphen inside alphanumeric address identifiers.
    s = ''.join(c if keep_character(c) or (
        address and c in '/-' and i > 0 and i + 1 < len(s)
        and s[i-1].isalnum() and s[i+1].isalnum()) else ' '
        for i, c in enumerate(s))
    return ' '.join(s.split())


def latin_fold(s):
    """Fold accents on LATIN letters only; never remove Indic vowel marks."""
    out = []
    latin = False
    for c in ud.normalize('NFD', s):
        if ud.category(c).startswith('M'):
            if not latin:
                out.append(c)
        else:
            latin = 'LATIN' in ud.name(c, '')
            out.append(c)
    return ud.normalize('NFC', ''.join(out))


def country_key(s):
    n = basic(s)
    return '' if unicode_normalize(s) in MISSING else COUNTRIES.get(n, n.upper())


def clean_name(raw):
    n = unicode_normalize(raw)
    missing = n in MISSING
    b = '' if missing else basic(raw)
    expanded = ' '.join(LEGAL.get(t, t) for t in basic(ID_TAG.sub(' ', raw)).split()) if not missing else ''
    tokens = expanded.split()
    while tokens and tokens[-1] in SUFFIXES:
        tokens.pop()
    core = ' '.join(tokens)
    return dict(name_raw=raw, name_normalized_unicode=n, name_basic=b,
        name_expanded=expanded, name_without_suffix=core,
        name_latin_folded=latin_fold(core), name_token_key=' '.join(sorted(set(tokens))),
        name_compact=''.join(tokens), name_missing=int(missing),
        name_had_id_tag=int(bool(ID_TAG.search(raw))))


def zero_normalize(s):
    return re.sub(r'\d+', lambda m: str(int(m[0])), s)


def clean_address(raw, country):
    # Remove only whole missing components, not words inside legitimate names.
    parts, removed = [], False
    for p in re.split(r'[,;|\n\r]+', raw):
        if unicode_normalize(p) in MISSING:
            removed = removed or bool(p.strip())
            continue
        q = re.sub(r'<\s*(?:null|none|na)\s*>', ' ', p, flags=re.I)
        removed = removed or q != p
        q = basic(q, address=True)
        if q:
            parts.append(q)
    missing = not parts
    state, state_index, postal = '', -1, ''
    for i, p in enumerate(parts):
        without_zip = re.sub(r'\b\d{5}(?:-\d{4})?\b', '', p).strip()
        if country == 'US':
            for code, full in US_STATES.items():
                if without_zip in (code.lower(), full) or re.search(r'\s' + code.lower() + r'$', without_zip):
                    state, state_index = code, i
        elif country == 'IN' and p in INDIA_STATES:
            state, state_index = p, i
    joined = ' '.join(parts)
    if country == 'IN':
        found = re.search(r'(?<!\w)[1-9]\d{5}(?!\w)', joined)
    elif country == 'US':
        # Require a state directly before the ZIP; avoids 5-digit house numbers.
        state_words = '|'.join(re.escape(x.lower()) for x in list(US_STATES) + list(US_STATES.values()))
        found = re.search(r'\b(?:' + state_words + r')\s+(\d{5}(?:-\d{4})?)(?!\w)', joined)
    elif country == 'FR':
        # Typical separate "75001 Paris" component; deliberately conservative.
        found = next((m for p in parts for m in [re.fullmatch(r'(\d{5})\s+[^\d]+', p)] if m), None)
    else:
        found = None
    if found:
        postal = found[1] if country in ('US', 'FR') else found[0]
    house = ''
    if parts:
        m = re.match(r'(?:(?:plot|house|door|no|number)\s+)?([a-z]?\d+[a-z]?(?:[-/][a-z0-9]+)*)\b', parts[0])
        if m and m[1] != postal:
            house = m[1]
    city = parts[state_index-1] if state_index > 0 else ''
    if parts and city == parts[0]:
        city = ''  # A street component must not be guessed to be a city.
    normalized_parts = []
    for i, p in enumerate(parts):
        tokens = p.split()
        for j, token in enumerate(tokens):
            if i == state_index and p in (state.lower(), US_STATES.get(state, '')):
                continue
            if country in ('US', 'IN'):
                if token in ROAD:
                    tokens[j] = ROAD[token]
                elif token in ('st', 'dr') and j >= 2 and any(c.isdigit() for c in tokens[0]):
                    # Avoid turning "St Louis" into "street louis".
                    tokens[j] = {'st': 'street', 'dr': 'drive'}[token]
                elif country == 'IN' and token == 'nr':
                    tokens[j] = 'near'
            elif country == 'FR' and token in ('av', 'bd'):
                tokens[j] = {'av': 'avenue', 'bd': 'boulevard'}[token]
        normalized_parts.append(' '.join(tokens))
    normalized = ' '.join(normalized_parts)
    numbers = sorted(set(t for t in joined.split() if any(c.isdigit() for c in t)))
    street = normalized_parts[0] if house else ''
    locality = next((p for p in normalized_parts if re.search(r'\b(nagar|colony|sector|phase|locality)\b', p)), '')
    return dict(address_raw=raw, address_basic=joined, address_normalized=normalized,
        address_latin_folded=latin_fold(normalized),
        address_tokens='|'.join(sorted(set(normalized.split()))),
        address_numbers='|'.join(numbers),
        address_numbers_zero_normalized='|'.join(sorted(set(map(zero_normalize, numbers)))),
        house_number_candidate=house, house_number_zero_normalized=zero_normalize(house),
        postal_code_candidate=postal, state_candidate=state, city_candidate=city,
        street_candidate=street, locality_candidate=locality,
        address_missing=int(missing), address_component_missing=int(removed))


def clean_record(row):
    country = country_key(row['country'])
    return dict(row, country_key=country, **clean_name(row['business_name']),
                **clean_address(row['business_address'], country))


def pair_features(a, b):
    """Prototype pair features. Missing evidence returns NaN, never a false zero.

    SequenceMatcher is intended for small candidate sets/prototyping, not an
    all-pairs comparison of millions of records. Retain all returned flags.
    """
    def similarity(x, y):
        return SequenceMatcher(None, x, y, autojunk=False).ratio() if x and y else math.nan
    def jaccard(x, y):
        x, y = set(x.split()), set(y.split())
        return len(x & y) / len(x | y) if x and y else math.nan
    return {
        'name_similarity': similarity(a['name_basic'], b['name_basic']),
        'name_expanded_similarity': similarity(a['name_expanded'], b['name_expanded']),
        'name_without_suffix_similarity': similarity(a['name_without_suffix'], b['name_without_suffix']),
        'name_token_jaccard': jaccard(a['name_without_suffix'], b['name_without_suffix']),
        'address_similarity': similarity(a['address_normalized'], b['address_normalized']),
        'address_token_jaccard': jaccard(a['address_normalized'], b['address_normalized']),
        'address_missing_left': int(a['address_missing']),
        'address_missing_right': int(b['address_missing']),
        'country_equal': int(a['country_key'] == b['country_key']) if a['country_key'] and b['country_key'] else math.nan,
        'house_number_equal': int(a['house_number_zero_normalized'] == b['house_number_zero_normalized']) if a['house_number_zero_normalized'] and b['house_number_zero_normalized'] else math.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('cleaned_data'))
    args = parser.parse_args()
    source, destination = args.input.resolve(), args.output.resolve()
    if not source.exists():
        parser.error('Input does not exist: ' + str(source))
    if destination.exists() and any(destination.iterdir()):
        parser.error('Output folder must be empty; choose a new folder to avoid overwriting results.')
    destination.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    entries = archive.namelist() if archive else [str(p.relative_to(source)) for p in source.rglob('*.tsv')]
    selected = sorted(n for n in entries if re.fullmatch(r'(train|test)_source[123]\.tsv', Path(n).name))
    if not selected:
        parser.error('No train_source1.tsv / test_source1.tsv etc. found in the input.')
    names = [Path(n).name for n in selected]
    if len(names) != len(set(names)):
        parser.error('Duplicate source filenames found; point --input to one dataset only.')
    def open_input(name):
        return archive.open(name) if archive else (source / name).open('rb')
    report = {'input': str(source), 'known_sample': (any(Path(n).name == 'SAMPLE_README.txt' for n in entries) if archive else any(source.rglob('SAMPLE_README.txt'))), 'files': {}, 'notes': [
        'All original field values and rows retained; no deduplication.',
        'Component columns are heuristic candidates; empty means unknown.',
        'Load TSVs with dtype=str, keep_default_na=False to preserve IDs and postal zeros.',
        'pair_features() calculates missing similarities as NaN after candidate generation.',
        'No transliteration: cross-script matches need additional retrieval/model features.']}
    try:
        for name in selected:
            split = Path(name).name.split('_')[0]
            target = destination / split / (Path(name).name + '.gz')
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.partial')
            counts = Counter()
            countries = Counter()
            with open_input(name) as binary, io.TextIOWrapper(binary, encoding='utf-8-sig', newline='') as inp, gzip.open(partial, 'wt', encoding='utf-8', newline='', compresslevel=1) as out:
                reader = csv.DictReader(inp, delimiter='\t')
                required = {'entity_id', 'business_name', 'business_address', 'country'}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f'{name}: missing required columns {required - set(reader.fieldnames or [])}')
                derived = set(clean_record(dict.fromkeys(required, ''))) - required
                if derived & set(reader.fieldnames):
                    raise ValueError(f'{name}: already contains cleaner output columns')
                writer = csv.DictWriter(out, fieldnames=reader.fieldnames + sorted(derived), delimiter='\t')
                writer.writeheader()
                for row in reader:
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError(f'{name}: malformed TSV row {reader.line_num}')
                    if not row['entity_id']:
                        raise ValueError(f'{name}: empty entity_id at row {reader.line_num}')
                    cleaned = clean_record(row)
                    writer.writerow(cleaned)
                    counts['rows'] += 1
                    for flag in ('name_missing', 'address_missing', 'address_component_missing'):
                        counts[flag] += cleaned[flag]
                    countries[cleaned['country_key']] += 1
                    if counts['rows'] % 100000 == 0:
                        print(f'{Path(name).name}: {counts["rows"]:,} rows', flush=True)
            partial.replace(target)
            report['files'][name] = dict(counts, countries=dict(countries), output=str(target))
            print(f'Completed {Path(name).name}: {counts["rows"]:,} rows', flush=True)
        truths = [n for n in entries if Path(n).name == 'train_ground_truth.tsv']
        if len(truths) > 1:
            raise ValueError('Multiple ground-truth files found')
        for name in truths:
            target = destination / 'train' / 'train_ground_truth.tsv'
            target.parent.mkdir(parents=True, exist_ok=True)
            with open_input(name) as inp, target.open('wb') as out:
                shutil.copyfileobj(inp, out, length=1024 * 1024)
        (destination / 'cleaning_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    finally:
        if archive:
            archive.close()
    print(f'Done. Output: {destination}')


if __name__ == '__main__':
    main()
