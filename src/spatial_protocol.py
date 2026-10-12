"""Physical block splits with a shared, label-independent query region."""
import numpy as np
from scipy.spatial import cKDTree


def common_query_mask(coordinates_um, candidate_blocks, buffer_um=500):
    coordinates_um=np.asarray(coordinates_um,dtype=float)
    if coordinates_um.ndim!=2 or coordinates_um.shape[1]!=2 or not np.isfinite(coordinates_um).all():
        raise ValueError('Coordinates must be finite N by2 micrometer values.')
    if buffer_um<0:raise ValueError('Negative buffer.')
    excluded=np.zeros(len(coordinates_um),dtype=bool)
    for block in candidate_blocks:
        rows=np.asarray(block,dtype=int)
        if not len(rows) or rows.min()<0 or rows.max()>=len(coordinates_um):
            raise ValueError('Invalid block indices.')
        distance=cKDTree(coordinates_um[rows]).query(coordinates_um,k=1)[0]
        excluded|=distance<=buffer_um
    return ~excluded


def assert_patient_disjoint(source_patients, target_patients):
    overlap=set(source_patients)&set(target_patients)
    if overlap:raise ValueError(f'Patient overlap: {sorted(overlap)}')
