import pandas as pd
data = pd.read_csv(r"data\cn15k\concept_id.csv", sep=',', encoding='utf-8')
data.to_csv(r"data\cn15k\entity_id.tsv", index=False, sep='\t', encoding='utf-8')