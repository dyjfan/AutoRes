import sys
import numpy as np
from pylab import show, plot
sys.path.append('../')
from pSCNN.snn import predict_pSCNN
from AutoRes.NetCDF import netcdf_reader 
from sklearn.metrics import explained_variance_score
from scipy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def back_remove(xx, point, range_point):
    xn = list(np.sum(xx, 1))
    n1 = xn.index(min(xn[0: range_point-point]))
    n3 = xn.index(min(xn[xx.shape[0]-range_point+point: xx.shape[0]]))
    if n1 < range_point-point/2:
        n2 = n1+3
    else:
        n2 = n1-3
    if n3 < xx.shape[0]-range_point-point/2:
        n4 = n3+3
    else:
        n4 = n3-3
    Ns = [[min(n1, n2), max(n1, n2)], [min(n3, n4), max(n3, n4)]]

    bak = np.zeros(xx.shape)
    for i in range(0, xx.shape[1]):
        tiab = []
        reg  = []
        for j in range(0, len(Ns)):
            tt = range(Ns[j][0], Ns[j][1])
            tiab.extend(xx[tt, i])
            reg.extend(np.arange(Ns[j][0], Ns[j][1]))
        rm = reg - np.mean(reg)
        tm = tiab - np.mean(tiab)
        b  = np.dot(np.dot(float(1)/np.dot(rm.T, rm), rm.T), tm)
        s  = np.mean(tiab) - np.dot(np.mean(reg), b)
        b_est = s + b * np.arange(xx.shape[0])
        bak[:, i] = xx[:, i] - b_est   
    bias = xx - bak
    return bak, bias

def plot_tic(re_X, xX, sta_C, sta_S):    
    plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.plot(np.sum(re_X, 1), label='re_chrom')
    ax1.plot(np.sum(xX, 1), label='actual_chrom')
    ax1.legend(fontsize=8)
    ax1.set_xlim([0,xX.shape[0]-1])
    ax1.get_yaxis().get_major_formatter().set_scientific(False)
    colors=['g','b','r']
    for i in range(len(sta_S)):
        name = 'compound'+str(i+1)
        ax2.plot(np.sum(np.dot(np.array(sta_C[:,i], ndmin=2).T,np.array(sta_S[i], ndmin=2)), 1), label=name,color=colors[i])
    ax2.legend(fontsize=8)
    ax2.set_xlim([0, xX.shape[0]-1])
    ax1.set_xlabel('Retention Time')
    ax1.set_ylabel('Intensity')
    ax1.set_ylim(bottom=0)
    ax2.set_xlabel('Scans')
    ax2.set_ylabel('Intensity')
    ax2.get_yaxis().get_major_formatter().set_scientific(False)
    ax2.set_ylim(bottom=0)
    x_major_locator=MultipleLocator(5)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax2.xaxis.set_major_locator(x_major_locator)
    plt.show()

def FRR(x, s, o, z, com):
    xs = x[s,:]   
    xs[xs<0]=0
    xz = x[z,:]
    xo = x[o,:]
    xc = np.vstack((xs, xz))
    mc = np.vstack((xs, np.zeros(xz.shape)))
    u, s0, v = np.linalg.svd(xc)
    t = np.dot(u[:,0:com],np.diag(s0[0:com]))
    r = np.dot(np.dot(np.linalg.pinv(np.dot(t.T, t)), t.T), np.sum(mc, 1))
    u1, s1, v1 = np.linalg.svd(x)
    t1 = np.dot(u1[:, 0:com], np.diag(s1[0:com]))
    c  = np.dot(t1, r)

    c1, ind = contrain_FRR(c, s, o)
    c1[c1<0]=0
    spec = x[s[ind],:]

    if c1[s[ind]] == 0:
        pu = 1e-6
    else:
        pu = c1[s[ind]]
        
    cc = c1/pu

    res_x = np.dot(np.array(cc, ndmin=2).T, np.array(spec, ndmin=2))
    left_x = x - res_x
    return cc, res_x

def contrain_FRR(c, s, m):
    ind_s = np.argmax(np.abs(c[s]))

    if c[s][ind_s] < 0:
        c = -c
    
    if s[0]<m[0]:
        if c[s[-2]] < c[s[-1]]:
            ind1 = s[-1]
            ind2 = m[np.argmax(c[m])]
        else:
            ind1 = s[np.argmax(c[s])]
            ind2 = m[0]
    else:
        if c[s[1]] < c[s[0]]:
            ind1 = m[np.argmax(c[m])]
            ind2 = s[0]
        else:
            ind1 = m[-1]
            ind2 = s[np.argmax(c[s])]

    for i, indd in enumerate(np.arange(ind1, 0, -1)):
        if c[indd-1] >= c[indd]:
            c[0:indd] = 0
            break
        if c[indd-1] < 0:
            c[0:indd] = 0
            break

    for i, indd in enumerate(np.arange(ind2, len(c)-1, 1)):
        if c[indd+1] >= c[indd]:
            c[indd+1]=0 
            break
        if c[indd+1] < 0:
            c[indd+1:len(c)] = 0
            break
    return c, ind_s

def predict_intervings(X, n, model1, model2):
    R = np.zeros_like(X)
    for i in range(len(X)):
        R[i] = X[n]
    Y1 = predict_pSCNN(model1, [R, X])
    Y2 = predict_pSCNN(model2, [R, X])
    in_1 = np.argwhere(Y1 > 0.51)[:,0]
    in_2 = np.argwhere(Y2 > 0.51)[:,0]
    ind_1 = best_index(X, n, in_1)
    ind_2 = best_index(X, n, in_2)
    return ind_1, ind_2

def best_index(X, n, index):
    ind_0 = np.zeros((0,), dtype=int)
    ind = np.zeros((0,), dtype=int)
    for i in sorted(set(index)):
        ind_0 = np.append(ind_0, i)
        if i+1 not in index:
            if len(ind_0) > len(ind):
                ind=ind_0
            ind_0 = np.zeros((0,), dtype=int)
    return ind

def max_r2_3(X, d_1, d_2, d_3, d_4, com):
    r_2 = 0
    for i in range(5):
        ind_1 = max(d_1)
        ind_1 = ind_1 + i
        for j in range(5):
            ind_2 = max(d_2)
            ind_2 = ind_2 + j
            if ind_1 < ind_2 and ind_1 < len(X) and ind_2 < len(X):
                s1 = list(range(0, ind_1+1))
                m1 = list(range(ind_1+1, ind_2+1))
                z1 = list(range(ind_2+1, X.shape[0]))
                cc1, xx1 = FRR(X, s1, m1, z1, com)
                for f in range(5):
                    ind_3 = min(d_3)
                    ind_3 = ind_3 - f
                    for t in range(5):
                        ind_4 = min(d_4)
                        ind_4 = ind_4 - t
                        if ind_3 > ind_4 and ind_3 > 0 and ind_4 > 0:
                            s3 = list(range(ind_3, X.shape[0]))
                            m3 = list(range(ind_4, ind_3))
                            z3 = list(range(0, ind_4))
                            cc3, xx3 = FRR(X, s3, m3, z3, com)
                        
                            xx2 = X-xx1-xx3
                            u, s, v = np.linalg.svd(xx2)
                            t = np.dot(u[:, 0:1], np.diag(s[0:1]))
                            cc2 = np.dot(np.dot(np.dot(t, np.linalg.pinv(np.dot(t.T, t))), t.T), np.sum(xx2,1 ))
                            cc2[cc2<0]=0
                            cc2 = np.array(cc2/norm(cc2), ndmin=2)
                            ss2 = np.dot(np.dot(np.linalg.pinv(np.dot(cc2, cc2.T)), cc2), xx2)
                            ss2[ss2<0]=0
                            
                            xx2 = np.dot(cc2.T, ss2)
                            re_x = xx1+xx2+xx3
                            R2 = explained_variance_score(X, re_x, multioutput='variance_weighted')    
                            if R2 > r_2:
                                r_2 = R2
                                n_1 = ind_1
                                n_2 = ind_2
                                n_3 = ind_3
                                n_4 = ind_4
    return r_2, n_1, n_2, n_3, n_4

def max_r2_2(X, d_1, d_2, com):
    r_2 = 0
    for i in range(5):
        ind_1 = max(d_1)
        ind_1 = ind_1 + i
        for j in range(5):
            ind_2 = max(d_2)
            ind_2 = ind_2 + j
            if ind_1 < ind_2 and ind_1 < len(X) and ind_2 < len(X):
                s1 = list(range(0, ind_1+1))
                m1 = list(range(ind_1+1, ind_2+1))
                z1 = list(range(ind_2+1, X.shape[0]))
                cc1, xx1 = FRR(X, s1, m1, z1, com)
                
                xx2 = X-xx1
                u, s, v = np.linalg.svd(xx2)
                t = np.dot(u[:, 0:1], np.diag(s[0:1]))
                cc2 = np.dot(np.dot(np.dot(t, np.linalg.pinv(np.dot(t.T, t))), t.T), np.sum(xx2, 1))
                cc2[cc2<0]=0
                cc2 = np.array(cc2/norm(cc2), ndmin=2)
                ss2 = np.dot(np.dot(np.linalg.pinv(np.dot(cc2, cc2.T)), cc2), xx2)
                ss2[ss2<0]=0
                
                xx2 = np.dot(cc2.T, ss2)
                re_x = xx1+xx2
                R2 = explained_variance_score(X, re_x, multioutput='variance_weighted')    
                if R2 > r_2:
                    r_2 = R2
                    n_1 = ind_1
                    n_2 = ind_2
    return r_2, n_1, n_2

def MCR_SNN(X, xX, new_num, model1, model2): 
    if len(new_num) > 3:
        print('Peak:'+str(len(new_num))+': The overlapping peak contains more than three components.')
    
    if len(new_num) == 1:
        u, s, v = np.linalg.svd(xX)
        t = np.dot(u[:,0:1], np.diag(s[0:1]))
        cc1 = np.dot(np.dot(np.dot(t, np.linalg.pinv(np.dot(t.T, t))), t.T), np.sum(xX, 1))
        cc1[cc1<0]=0
        cc1 = np.array(cc1/norm(cc1), ndmin=2)
        ss1 = np.dot(np.dot(np.linalg.pinv(np.dot(cc1, cc1.T)), cc1), xX)
        ss1[ss1<0]=0
        
        xx1 = np.dot(cc1.T, ss1)
        r2 = explained_variance_score(xX, xx1, multioutput='variance_weighted')    
        
        
        sta_S = np.zeros_like(ss1)
        sta_S[0] = ss1/np.max(ss1)*999
        sta_C = np.dot(np.dot(xX, sta_S.T), np.linalg.pinv(np.dot(sta_S, sta_S.T)))
        sta_C[sta_C<0]=0
        re_X=np.dot(sta_C,sta_S)
        R2 = explained_variance_score(xX, re_X, multioutput='variance_weighted')
    
    if len(new_num) == 2:
        R0 = np.zeros_like(X)
        for i in range(len(X)):
            R0[i] = X[0]
        Y0 = predict_pSCNN(model1, [R0, X])
        ind_0 = np.argwhere(Y0 < 0.8)[:, 0]
        
        ind_1, ind_2 = predict_intervings(X, min(ind_0)+4, model1, model2)
        r_2, n_1, n_2 = max_r2_2(X, ind_1, ind_2, len(new_num))
        s1 = list(range(0, n_1+1))
        m1 = list(range(n_1+1, n_2+1))
        z1 = list(range(n_2+1, X.shape[0]))
        cc1, xx1 = FRR(X, s1, m1, z1, len(new_num))  
        xx2 = X-xx1
        
        u, s, v = np.linalg.svd(xx2)
        t = np.dot(u[:,0:1], np.diag(s[0:1]))
        cc2 = np.dot(np.dot(np.dot(t, np.linalg.pinv(np.dot(t.T, t))), t.T), np.sum(xx2,1))
        cc2[cc2<0]=0
        cc2 = np.array(cc2/norm(cc2), ndmin=2)
        ss2 = np.dot(np.dot(np.linalg.pinv(np.dot(cc2, cc2.T)), cc2), xx2)
        ss2[ss2<0]=0
        
        xx2 = np.dot(cc2.T, ss2)
        re_x = xx1+xx2
        r2 = explained_variance_score(X, re_x, multioutput='variance_weighted')    
        
        re_C = np.vstack([np.array(cc1, ndmin=2), cc2]).T
        S1 = np.dot(np.dot(np.linalg.pinv(np.dot(re_C.T, re_C)), re_C.T), xX)
        for i in range(0, 200):
            S1[S1<0]=0
            C1 = np.dot(np.dot(xX,S1.T), np.linalg.pinv(np.dot(S1, S1.T)))
            C1[C1<0]=0
            S1 = np.dot(np.dot(np.linalg.pinv(np.dot(C1.T, C1)), C1.T), xX)  
        
        sta_S = np.zeros_like(S1)
        sta_S[0] = S1[0]/np.max(S1[0])*999
        sta_S[1] = S1[1]/np.max(S1[1])*999
        sta_C = np.dot(np.dot(xX, sta_S.T), np.linalg.pinv(np.dot(sta_S, sta_S.T)))
        sta_C[sta_C<0]=0
        re_X=np.dot(sta_C, sta_S)
        R2 = explained_variance_score(xX, re_X, multioutput='variance_weighted')
    
    if len(new_num) == 3:
        R0 = np.zeros_like(X)
        for i in range(len(X)):
            R0[i] = X[0]
        Y0 = predict_pSCNN(model1, [R0, X])
        ind_0 = np.argwhere(Y0 < 0.8)[:, 0]
        
        ind_1, ind_2 = predict_intervings(X, min(ind_0)+4, model1, model2)
        ind_3, ind_4 = predict_intervings(X, max(ind_0)-4, model1, model2)
        r_2, n_1, n_2, n_3, n_4 = max_r2_3(X, ind_1, ind_2, ind_3, ind_4, len(new_num))
        
        s1 = list(range(0, n_1+1))
        m1 = list(range(n_1+1, n_2+1))
        z1 = list(range(n_2+1, X.shape[0]))
        cc1, xx1 = FRR(X, s1, m1, z1, len(new_num))
        s3 = list(range(n_3, X.shape[0]))
        m3 = list(range(n_4, n_3))
        z3 = list(range(0, n_4))
        cc3, xx3 = FRR(X, s3, m3, z3, len(new_num))
        xx2= X-xx1-xx3
        
        u, s, v = np.linalg.svd(xx2)
        t = np.dot(u[:, 0:1], np.diag(s[0:1]))
        cc2 = np.dot(np.dot(np.dot(t, np.linalg.pinv(np.dot(t.T, t))), t.T), np.sum(xx2,1))
        cc2[cc2<0]=0
        cc2 = np.array(cc2/norm(cc2), ndmin=2)
        ss2 = np.dot(np.dot(np.linalg.pinv(np.dot(cc2, cc2.T)), cc2), xx2)
        ss2[ss2<0]=0
        
        xx2 = np.dot(cc2.T, ss2)
        re_x = xx1+xx2+xx3
        r2 = explained_variance_score(X, re_x, multioutput='variance_weighted')
        
        
        re_C = np.vstack([np.array(cc1, ndmin=2), cc2, np.array(cc3, ndmin=2)]).T
        S1 = np.dot(np.dot(np.linalg.pinv(np.dot(re_C.T, re_C)), re_C.T), xX)
        for i in range(0, 200):
            S1[S1<0]=0
            C1 = np.dot(np.dot(xX,S1.T),np.linalg.pinv(np.dot(S1, S1.T)))
            C1[C1<0]=0
            S1 = np.dot(np.dot(np.linalg.pinv(np.dot(C1.T, C1)), C1.T), xX)
        
        sta_S = np.zeros_like(S1)
        sta_S[0] = S1[0]/np.max(S1[0])*999
        sta_S[1] = S1[1]/np.max(S1[1])*999
        sta_S[2] = S1[2]/np.max(S1[2])*999
        sta_C = np.dot(np.dot(xX, sta_S.T), np.linalg.pinv(np.dot(sta_S, sta_S.T)))
        sta_C[sta_C<0]=0
        re_X=np.dot(sta_C, sta_S)
        R2 = explained_variance_score(xX, re_X, multioutput='variance_weighted')
    return sta_S, sta_C, re_X, r2, R2

def AutoRes(filename, model1, model2):
    ncr = netcdf_reader(filename, True)
    m = np.array(ncr.mat(0, len(ncr.tic()['rt'])-1).T, dtype='float32')
    ms = np.array(ncr.mz_rt(10)['mz'], dtype='int')
    mz_range = (1, 1000)
    mz_min, mz_max = mz_range
    mz_dense = np.linspace(int(mz_min), int(mz_max), int(mz_max-mz_min)+1, 
                           dtype=np.float32)
    X = np.zeros((m.shape[0], mz_max), dtype = np.float32)
    for i in range(m.shape[0]):
        itensity_dense = np.zeros_like(mz_dense)
        itensity_dense[ms-mz_min] = m[i]
        X[i] = itensity_dense
        X[i][X[i]<0]=0
    num = len(X)//500
    io = np.empty(0, dtype='int32')
    for i in range(num-1):
        X0=X[500*i:500*(i+1)]
        ind = np.arange(500*i,500*(i+1),1, dtype='int32')
        xX, bias = back_remove(X0, 4, 10)
        xX[xX<0]=0
        noise=np.mean(np.sort(np.sum(xX, 1))[0:300])
        ind_0 = np.argwhere(np.sum(xX, 1) >= 3*noise)[:, 0]
        io = np.hstack((io, ind[ind_0]))
    X0=X[500*(num-1):len(X)]
    ind = np.arange(500*(num-1),len(X),1, dtype='int32')
    xX, bias = back_remove(X0, 4, 10)
    xX[xX<0]=0
    noise=np.mean(np.sort(np.sum(xX, 1))[0:300])
    ind_1 = np.argwhere(np.sum(xX, 1) >= 3*noise)[:, 0]
    io = np.hstack((io, ind[ind_1]))
    l= []
    ls= []
    for x in io:
        l.append(x)
        if x+1 not in io:
           if len(l)>=7:
               ls.append(l)
           l=[]
    ls[0]=list(range(ls[0][0]-5,ls[0][-1]+1)) 
    ls[-1]=list(range(ls[-1][0],ls[-1][-1]+6))
    for i in range(len(ls)-1):
        if ls[i+1][0]-ls[i][-1]>=8:
            ls[i]=list(range(ls[i][0],ls[i][-1]+6))
            ls[i+1]=list(range(ls[i+1][0]-5,ls[i+1][-1]+1))    
            
    sta_S0=np.empty((0,1000), dtype='float32')
    rt=[]
    area=[]
    r_2_0=[]
    r_2_1=[]
    for j in range(len(ls)):
        m = np.array(ncr.mat(ls[j][0], ls[j][-1]).T, dtype='float32')
        if len(ls[j])<=20 and np.sum(m,1)[0]>=1.02*np.sum(m,1)[-1]:
            m0 = np.array(ncr.mat(ls[j][0]-4, ls[j][-1]+1).T, dtype='float32')
            t0=np.arange(ls[j][0]-4,ls[j][-1]+2)
        else:
            m0 = np.array(ncr.mat(ls[j][0]+1, ls[j][-1]).T, dtype='float32')
            t0=np.arange(ls[j][0]+1,ls[j][-1])
        m1 = np.array(ncr.mat(ls[j][0]-1, ls[j][-1]).T, dtype='float32')
        t1=np.arange(ls[j][0]-1,ls[j][-1]+1)
        def converts(m):
            X = np.zeros((m.shape[0], mz_max), dtype = np.float32)
            for i in range(m.shape[0]):
                itensity_dense = np.zeros_like(mz_dense)
                itensity_dense[ms-mz_min] = m[i]
                X[i] = itensity_dense
                X[i][X[i]<0]=0
            xX, bias = back_remove(X, 4, 10)
            xX[xX<0]=0
            X=xX/np.max(xX)
            return X, xX
        X0, xX0 = converts(m0)
        X1,xX1 = converts(m1)
        u, s0, v = np.linalg.svd(X0)
        new_num=s0[s0>2*np.mean(s0)]
        if len(new_num)>1:
            try:
                S_0, C_0, re_X_0, r2_0, R2_0 = MCR_SNN(X0, xX0, new_num, model1, model2)
            except:
                try:
                    S_0, C_0, re_X_0, r2_0, R2_0 = MCR_SNN(X0, xX0, new_num[0:-1], model1, model2)
                except:
                    S_0, C_0, re_X_0, r2_0, R2_0 = MCR_SNN(X0, xX0, new_num[0:-1][0:-1], model1, model2)
            try:
                S_1, C_1, re_X_1, r2_1, R2_1 = MCR_SNN(X1, xX1, new_num, model1, model2)
            except:
                try:
                    S_1, C_1, re_X_1, r2_1, R2_1 = MCR_SNN(X1, xX1, new_num[0:-1], model1, model2)
                except:
                    S_1, C_1, re_X_1, r2_1, R2_1 = MCR_SNN(X1, xX1, new_num[0:-1][0:-1], model1, model2)
            if r2_0 > r2_1:
                sta_S, sta_C, re_X, r2, R2 = S_0, C_0, re_X_0, r2_0, R2_0
                t=t0
                X = X0
                xX=xX0
            else:
                sta_S, sta_C, re_X, r2, R2 = S_1, C_1, re_X_1, r2_1, R2_1
                t=t1
                X = X1
                xX=xX1
       
        if len(new_num)==1:
            sta_S, sta_C, re_X, r2, R2= MCR_SNN(X0, xX0, new_num, model1, model2)
            t=t0
            X = X0
            xX=xX0
       
        #plot_tic(re_X, xX, sta_C, sta_S)
        sta_S0 = np.vstack((sta_S0,sta_S))
        for i in range(len(sta_S)):
            r_2_0.append(r2)
            r_2_1.append(R2)
        for i in range(len(sta_S)):
            maxindex  = np.argmax(sta_C[:,i])
            tic=ncr.tic()
            rt0=round(tic['rt'][t[maxindex]].astype(np.float32), 2)
            rt.append(rt0)
            compound=np.trapz(np.sum(np.dot(np.array(sta_C[:,i], ndmin=2).T,np.array(sta_S[i], ndmin=2)), 1, dtype='float32'))
            area.append(compound)
    return sta_S0, area, rt, r_2_1

def output_msp(filename, sta_S, RT):
    sta_S[sta_S<3]=0
    f = open(filename, "x")
    for i in range(len(sta_S)):
        f.write("Name: ")
        f.write(str(RT[i]))
        f.write('\n')
        f.write("RT: ")
        f.write(str(RT[i]))
        f.write('\n')
        f.write("Num Peaks: ")
        f.write(str(sta_S.shape[1]))
        f.write('\n')
        for n in range(sta_S.shape[1]):
            f.write(str(n+1))
            f.write(' ')
            f.write(str(sta_S[i,n]))
            f.write('\n')
        f.write('\n')
    f.close()
