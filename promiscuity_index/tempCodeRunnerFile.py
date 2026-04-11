import pandas as pd
import numpy as np 
import os 

curr_path = os.getcwd()
print(curr_path)

drug_target_intertaction = pd.read_csv('raw_data/drug.target.interaction.tsv.gz',sep="\t", compression="gzip")

print(drug_target_intertaction.sample(5))