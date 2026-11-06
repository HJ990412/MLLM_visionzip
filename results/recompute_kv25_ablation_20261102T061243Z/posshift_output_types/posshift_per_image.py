# Per-image consistency of PosShift hit outputs (table 5); input is posshift_hits.json.
import json, sys, collections
rows = json.load(open(sys.argv[1]))
img = collections.defaultdict(list)
for r in rows: img[r['image_id']].append(r['prediction'].strip() == '')
c = collections.Counter(sum(v) for v in img.values()); print('images by #blank outputs (of 5):', sorted(c.items()))
print('distinct exact outputs per image:', sorted(collections.Counter(len(set(r['prediction'] for r in rows if r['image_id'] == i)) for i in img).items()))
