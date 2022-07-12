import sqlite3 as sqlite
import numpy as np
import json, sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def get_spectra_sqlite(dbname):
    spectra = []  
    conn = sqlite.connect(dbname)
    conn.row_factory = sqlite.Row
    rows=conn.cursor().execute('SELECT * FROM spectra').fetchall()
    for row in tqdm(rows, desc="Get spectra"):
        s = {}
        s['name']      = row['name']
        s['RI']        = row['RI']
        s['mz']        = np.array(json.loads(row['mz']), dtype = np.integer)
        s['intensity'] = np.array(json.loads(row['intensity']), dtype=np.float32)
        spectra.append(s)
    conn.close()
    return spectra

def save_spectra_sqlite(dbname, spectra):
    conn = sqlite.connect(dbname)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS spectra")
    cur.execute("CREATE TABLE spectra (name TEXT, RI TEXT, mz TEXT, intensity TEXT, PRIMARY KEY(name))")
    for s in tqdm(spectra, desc="Save spectra"):
        cur.execute(f"insert into spectra (name, RI, mz, intensity) values \
                    (\"{s['name']}\", \"{s['RI']}\", \"{s['mz'].tolist()}\", \"{s['intensity'].tolist()}\")")
    conn.commit()
    conn.close()

def filter_spectra(spectra, mz_range):
    spectra_filtered = []
    for s in spectra:
        if s['mz'][0]>= mz_range[0] and s['mz'][-1]<=mz_range[1]:
            spectra_filtered.append(s)
    return spectra_filtered
    
def rand_sub_sqlite(spectra, dbn, n):
    ids = np.random.permutation(len(spectra))[0:n]
    sn  = [spectra[i] for i in ids]
    save_spectra_sqlite(dbn, sn)
    
def rand_sub_sqlite1(spectra, dbn, n1, n2):
    sn  = [spectra[i] for i in range(n1,n2)]
    save_spectra_sqlite(dbn, sn)
    
def get_mz_ranges(spectra):
    mz_mins = []
    mz_maxs = []
    for s in tqdm(spectra, desc='Get mz range'):
        mz_mins.append(s['mz'][0])
        mz_maxs.append(s['mz'][-1])
    return mz_mins, mz_maxs

def plot_mz_hist(mz_range):
    plt.hist(mz_range[0], 1000)
    plt.figure()
    plt.hist(mz_range[1], 1000)

def convert_to_dense(spectra, mz_range):
    mz_min, mz_max = mz_range
    mz_dense = np.linspace(int(mz_min), int(mz_max), int(mz_max-mz_min)+1, dtype=np.float32)
    for s in tqdm(spectra, desc='Convert to dense'):
        itensity_dense = np.zeros_like(mz_dense)
        itensity_dense[s['mz']-mz_min] = s['intensity']
        s['mz_dense'] = mz_dense
        s['intensity_dense'] = itensity_dense/np.max(itensity_dense)
        
def sims(dbname, mz_range):
    spectra = get_spectra_sqlite(dbname)
    convert_to_dense(spectra, mz_range)
    c=[]
    for i in range(len(spectra)):
        com1_R = np.array(spectra[i]['intensity_dense'], ndmin=2, dtype = np.float32)
        sim=[]
        for j in range(len(spectra)):
            com1_S = np.array(spectra[j]['intensity_dense'], ndmin=2, dtype = np.float32)
            sim0 = cosine_similarity(com1_R, com1_S)
            sim.append(sim0[0,0])
        ind = [i]
        for num in sim:
            if 0.8 < num and num < 0.9:
                ind.append(sim.index(num))
        if len(ind) > 4:
            c.append(ind)
    return c
