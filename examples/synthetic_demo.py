"""Synthetic pipeline check. Output is NOT an experimental research result."""
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation import mean_gene_pcc
from src.spatial_protocol import common_query_mask,assert_patient_disjoint
from src.interval_math import split_quantile,selftest


def main():
    rng=np.random.default_rng(20261012)
    source_x=rng.normal(size=(400,16));beta=rng.normal(size=(16,4))
    source_y=source_x@beta+rng.normal(scale=.4,size=(400,4))
    xx,yy=np.meshgrid(np.arange(40),np.arange(40));coordinates=np.column_stack([xx.ravel(),yy.ravel()])*100
    target_x=rng.normal(size=(len(coordinates),16))+1
    target_y=target_x@(beta+.2)+rng.normal(scale=.6,size=(len(coordinates),4))
    blocks=[np.flatnonzero((xx.ravel()>=a)&(xx.ravel()<a+5)&(yy.ravel()>=b)&(yy.ravel()<b+10)) for a,b in [(0,0),(15,15)]]
    query=common_query_mask(coordinates,blocks,buffer_um=500)
    assert_patient_disjoint(['synthetic-source'],['synthetic-target'])
    assert not np.intersect1d(np.flatnonzero(query),np.concatenate(blocks)).size
    scaler=StandardScaler().fit(source_x);z=scaler.transform(source_x)
    model=Ridge(alpha=10).fit(z,source_y)
    annotation=blocks[0]
    adapted=Ridge(alpha=10).fit(np.vstack([z,scaler.transform(target_x[annotation])]),
        np.vstack([source_y,target_y[annotation]]),sample_weight=np.r_[np.full(len(z),.5/len(z)),np.full(len(annotation),.5/len(annotation))])
    pred=adapted.predict(scaler.transform(target_x[query]))
    # Use a separate frozen source predictor for this simple split-CP illustration.
    residual=np.abs(target_y[annotation]-model.predict(scaler.transform(target_x[annotation])))
    q=split_quantile(residual,.9)
    frozen_query=model.predict(scaler.transform(target_x[query]))
    coverage=float((np.abs(target_y[query]-frozen_query)<=q).mean())
    print(json.dumps(dict(kind='synthetic_workflow_check_not_scientific_result',query_spots=int(query.sum()),
        annotation_spots=len(annotation),demo_adapted_PCC=mean_gene_pcc(pred,target_y[query]),
        empirical_split_coverage=coverage,theoretical_exchangeability_not_assumed=True,
        interval_math_checks=selftest()),indent=2))


if __name__=='__main__':main()
