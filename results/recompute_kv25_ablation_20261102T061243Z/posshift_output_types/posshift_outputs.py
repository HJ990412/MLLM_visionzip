import json, sys, collections
raw = sys.argv[1]
rows = [json.loads(l) for l in open(raw)]
arms = collections.Counter(r['arm'] for r in rows)
print('arms', dict(arms))
by = {(r['image_id'], r['turn'], r['arm']): r for r in rows if r['status'] == 'OK'}
ps = [r for r in rows if r['arm'] == 'posshift' or 'shift' in r['arm']]
print('posshift rows', len(ps), collections.Counter((r['turn'], r['cache_hit'], r.get('executed')) for r in ps))
hit = [r for r in ps if r['cache_hit']]
pred = collections.Counter(r['prediction'] for r in hit)
print('distinct predictions', len(pred))
for p, c in pred.most_common(40): print(f'{c:4d} | {p!r}')
print('first tokens', collections.Counter(r['first_token_id'] for r in hit).most_common(15))
print('lengths', collections.Counter(len(r['generated_token_ids']) for r in hit).most_common())
print('acc', sum(r['accuracy'] for r in hit))
json.dump([{k: r[k] for k in ('image_id','turn','question','gold','prediction','first_token_id','generated_token_ids')}
           | {'ours_prediction': by.get((r['image_id'], r['turn'], 'ours'), {}).get('prediction')} for r in hit],
          open(sys.argv[2], 'w'), ensure_ascii=False, indent=0)
