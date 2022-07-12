import random
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

def augment_positive(spectra, ids4p, low=0.1, high=1.0):
    no = random.randint(len(ids4p)-1, len(ids4p)-1)
    ratio = random.uniform(low, high)
    x = ratio * spectra[ids4p[-1]]['intensity_dense']
    for i in range(no):
        ratio = random.uniform(low, high)
        x = x + ratio * spectra[ids4p[i]]['intensity_dense']
    return x

def augment_negative(spectra, ids4n, low=0.1, high=1.0):
    no = random.randint(len(ids4n)-1, len(ids4n)-1)
    x = np.zeros_like(spectra[0]['intensity_dense'])
    for i in range(no):
        ratio = random.uniform(low, high)
        x = x + ratio * spectra[ids4n[i]]['intensity_dense']
    return x

def data_augmentation_1(spectra, n, maxn, noise_level=0.001):
    p  = spectra[0]['mz_dense'].shape[0]
    s  = len(spectra)
    Rp = np.zeros((n, p), dtype = np.float32)
    Sp = np.zeros((n, p), dtype = np.float32)
    Rn = np.zeros((n, p), dtype = np.float32)
    Sn = np.zeros((n, p), dtype = np.float32)
    for i in tqdm(range(n), desc="Data augmentation"):
        n1 = np.random.normal(0, 1, p)
        n2 = np.random.normal(0, 1, p)
        n3 = np.random.normal(0, 1, p)
        n4 = np.random.normal(0, 1, p)
        ids4p   = random.sample(range(0, s-1), maxn)
        Rp[i, ] = spectra[ids4p[-1]]['intensity_dense'] + (n1-np.min(n1))*noise_level
        Sp[i, ] = augment_positive(spectra, ids4p) + (n2-np.min(n2))*noise_level
        ids4n   = random.sample(range(0, s-1), maxn+1)
        Rn[i, ] = spectra[ids4n[-1]]['intensity_dense'] + (n3-np.min(n3))*noise_level
        Sn[i, ] = augment_negative(spectra, ids4n) + (n4-np.min(n4))*noise_level
    R = np.vstack((Rp, Rn))
    S = np.vstack((Sp, Sn))
    y = np.concatenate ((np.ones(n, dtype = np.float32), np.zeros(n, dtype = np.float32)), axis=None)
    R, S, y = shuffle(R, S, y)
    return {'R':R, 'S':S, 'y':y}

def data_augmentation_2(spectra, c, n, m, maxn, noise_level=0.001):
    p  = spectra[0]['mz_dense'].shape[0]
    s  = len(c)
    Rp0 = np.zeros((n, p), dtype = np.float32)
    Sp0 = np.zeros((n, p), dtype = np.float32)
    Rn0 = np.zeros((n, p), dtype = np.float32)
    Sn0 = np.zeros((n, p), dtype = np.float32)
    for i in tqdm(range(n), desc="Data augmentation"):
        n1 = np.random.normal(0, 1, p)
        n2 = np.random.normal(0, 1, p)
        n3 = np.random.normal(0, 1, p)
        n4 = np.random.normal(0, 1, p)
        P        = random.sample(range(0, s-1), 1)
        ids4p    = random.sample(c[P[0]][1:-1], maxn-1)
        ids4p.append(c[P[0]][0])
        Rp0[i, ] = spectra[ids4p[-1]]['intensity_dense'] + (n1-np.min(n1))*noise_level
        Sp0[i, ] = augment_positive(spectra, ids4p) + (n2-np.min(n2))*noise_level
        N        = random.sample(range(0, s-1), 1)
        ids4n    = random.sample(c[N[0]][1:-1], maxn)
        ids4n.append(c[N[0]][0])
        Rn0[i, ] = spectra[ids4n[-1]]['intensity_dense'] + (n3-np.min(n3))*noise_level
        Sn0[i, ] = augment_negative(spectra, ids4n) + (n4-np.min(n4))*noise_level
    R0 = np.vstack((Rp0, Rn0))
    S0 = np.vstack((Sp0, Sn0))
    y0 = np.concatenate ((np.ones(n, dtype = np.float32), np.zeros(n, dtype = np.float32)), axis=None)
    p  = spectra[0]['mz_dense'].shape[0]
    s  = len(spectra)
    Rp1 = np.zeros((m, p), dtype = np.float32)
    Sp1 = np.zeros((m, p), dtype = np.float32)
    Rn1 = np.zeros((m, p), dtype = np.float32)
    Sn1 = np.zeros((m, p), dtype = np.float32)
    for i in tqdm(range(m), desc="Data augmentation"):
        n1 = np.random.normal(0, 1, p)
        n2 = np.random.normal(0, 1, p)
        n3 = np.random.normal(0, 1, p)
        n4 = np.random.normal(0, 1, p)
        ids4p    = random.sample(range(0, s-1), maxn)
        Rp1[i, ] = spectra[ids4p[-1]]['intensity_dense'] + (n1-np.min(n1))*noise_level
        Sp1[i, ] = augment_positive(spectra, ids4p) + (n2-np.min(n2))*noise_level
        ids4n    = random.sample(range(0, s-1), maxn+1)
        Rn1[i, ] = spectra[ids4n[-1]]['intensity_dense'] + (n3-np.min(n3))*noise_level
        Sn1[i, ] = augment_negative(spectra, ids4n) + (n4-np.min(n4))*noise_level
    R1 = np.vstack((Rp1, Rn1))
    S1 = np.vstack((Sp1, Sn1))
    y1 = np.concatenate ((np.ones(m, dtype = np.float32), np.zeros(m, dtype = np.float32)), axis=None)
    R  = np.vstack((R0, R1))
    S  = np.vstack((S0, S1))
    y  = np.hstack((y0, y1))
    R, S, y = shuffle(R, S, y)
    return {'R':R, 'S':S, 'y':y}
