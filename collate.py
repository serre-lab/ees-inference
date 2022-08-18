import json,os,glob
from pathlib import Path

BASE_PATH = '/media/data_cifs/projects/prj_deepspine/minju/deep-spine-ignite'
EXP_PATH = [
    'multirun/2020-12-17/19-44-12', 
    'multirun/2020-12-17/19-54-55', 
    'multirun/2020-12-17/20-02-37',
    'multirun/2020-12-17/20-05-45',
    'multirun/2020-12-17/20-14-14'
] #sweep_output_dir

files = []
for exp_path in EXP_PATH:
    for path in Path(os.path.join(BASE_PATH, exp_path)).rglob('proposals.json'):                                                   
        files.append(path)

output_filename = 'collated_proposals.json'
if os.path.exists(output_filename):
    os.remove(output_filename)

cjson = open(output_filename, 'a', encoding='utf-8')

for f in files:
    X = json.load(open(f,'r'))
    json.dump(X, cjson, ensure_ascii=False, indent='\t')
