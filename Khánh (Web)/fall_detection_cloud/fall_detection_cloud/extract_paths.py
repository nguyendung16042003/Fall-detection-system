import json

data = json.load(open('openapi.json', encoding='utf-8'))
paths = data['paths']
for k in sorted(paths.keys()):
    methods = list(paths[k].keys())
    print(f'{k}: {methods}')
