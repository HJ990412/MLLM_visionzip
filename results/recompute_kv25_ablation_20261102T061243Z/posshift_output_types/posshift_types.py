import json, sys, re, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('llava-hf/llava-v1.6-vicuna-7b-hf')
rows = json.load(open(sys.argv[1]))
def kind(r):
    p = r['prediction']; ids = r['generated_token_ids']
    if p.strip() == '': return 'A 공백/개행만'
    words = re.findall(r'[A-Za-z]+', p)
    if words and set(w.lower() for w in words) <= {'in','on','to','with','is','the','s','at','of','a'}: return 'C 전치사·기능어 반복'
    if not words or re.fullmatch(r'[\s|\[\]=:﻿]*', p): return 'B 기호 반복 (| [[ ==)'
    if re.search(r'd:|pol:|c:', p): return 'D 기호+조각 (d: pol: …)'
    return 'E 기타 혼합'
groups = collections.defaultdict(list)
for r in rows: groups[kind(r)].append(r)
for k in sorted(groups):
    g = groups[k]
    print(f'== {k}: {len(g)}  turns {dict(sorted(collections.Counter(r["turn"] for r in g).items()))}')
    print('   first tok', [(tok.convert_ids_to_tokens(t), c) for t, c in collections.Counter(r['first_token_id'] for r in g).most_common(4)])
    for r in g[:3]:
        print('   Q:', r['question'], '| gold:', r['gold'], '| Ours:', repr(r['ours_prediction']))
        print('     tokens:', tok.convert_ids_to_tokens(r['generated_token_ids']))
eos = sum(2 in r['generated_token_ids'] for r in rows)
print('EOS present', eos, '/', len(rows))
for t in range(2, 7):
    g = [r for r in rows if r['turn'] == t]
    print(t, collections.Counter(kind(r)[0] for r in g))
