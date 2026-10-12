"""Read the exported completed results; no fitting or new statistical tests."""
import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with (root/'results/model_annotation_2x2.csv').open(encoding='utf-8') as f:
    for row in csv.DictReader(f):
        print(json.dumps({k:row[k] for k in ['cancer','n_independent_target_patients','n_genes','Ridge0_mean_PCC','Ridge200_mean_PCC','TRIPLEX200_mean_PCC']},ensure_ascii=False))
