import re
from ast import literal_eval
import pandas as pd

## SCRIPT FOR CONVERTING HPOS FROM 2nd HYPERPARAMETER SEARCH (log/hpo_2nd.log) TO CSV


with open('/work/aicos/output/hpo2.txt', 'r') as f: # hpo2.txt contains the printout from log/hpo_2nd.log
    lines = f.read()

obj_re = re.compile(r'(?<=Objective value: )\d\.\d+')
hpos_re = re.compile(r'(?<=Hyperparameters: ).*?(?=\n|$)')

hpos = hpos_re.findall(lines)
hpos = [literal_eval(hpo) for hpo in hpos]
objvs = obj_re.findall(lines)

for i, hpo in enumerate(hpos):

    hpo['objective_value'] = float(objvs[i])

hpo_df = pd.DataFrame.from_records(hpos)

hpo_df.to_csv('aicos_hpo.csv', index = False)