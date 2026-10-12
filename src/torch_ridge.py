"""Source-fitted float64 Ridge excerpt from the actual GPU experiment pipeline.

Requires optional PyTorch; accepts CPU or CUDA devices. No private paths.
"""

def model_fit(x, y, device, alpha=1000):
    """StandardScaler + centered float64 Ridge, identical sum-loss alpha."""
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    mean = y.mean(0)
    gram = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float64, device=device)
    cross = torch.zeros((x.shape[1], y.shape[1]), dtype=torch.float64, device=device)
    for first in range(0, len(x), 4096):
        gx = torch.as_tensor(z[first:first + 4096], dtype=torch.float64, device=device)
        gy = torch.as_tensor(y[first:first + 4096] - mean, dtype=torch.float64, device=device)
        gram += gx.T @ gx
        cross += gx.T @ gy
    beta = torch.cholesky_solve(cross, torch.linalg.cholesky(gram + alpha * torch.eye(x.shape[1], dtype=torch.float64, device=device)))
    return dict(beta=beta.cpu().numpy(), intercept=mean, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)

def predict(model, x):
    return (x - model['scaler_mean']) / model['scaler_scale'] @ model['beta'] + model['intercept']
