"""Patient-level summaries and conservative mean-gene PCC."""
import numpy as np


def mean_gene_pcc(prediction, truth):
    prediction=np.asarray(prediction,dtype=float);truth=np.asarray(truth,dtype=float)
    if prediction.shape!=truth.shape or prediction.ndim!=2:
        raise ValueError('Expected matched spots by genes arrays.')
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite input.')
    p=prediction-prediction.mean(0);y=truth-truth.mean(0)
    denominator=np.sqrt((p*p).sum(0)*(y*y).sum(0))
    values=np.divide((p*y).sum(0),denominator,out=np.zeros(truth.shape[1]),where=denominator>0)
    values[np.ptp(truth,axis=0)==0]=np.nan
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def paired_patient_interval(differences, repetitions=10000, seed=20261010):
    """Input contains ONE already-averaged difference per patient."""
    values=np.asarray(differences,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Require finite patient-level differences.')
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(repetitions,len(values)))].mean(1)
    lo,hi=np.quantile(means,[.025,.975])
    return dict(mean=float(values.mean()),CI_low=float(lo),CI_high=float(hi),n_patients=len(values))
