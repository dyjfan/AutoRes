# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:51:13 2020

@author: zmzhang
"""
import pickle
from pSCNN.db import get_spectra_sqlite, save_spectra_sqlite, rand_sub_sqlite, rand_sub_sqlite1, \
                      get_mz_ranges, convert_to_dense, plot_mz_hist, filter_spectra
from pSCNN.da import data_augmentation_1, data_augmentation_2
from pSCNN.snn import plot_loss_accuracy, check_pSCNN, build_pSCNN, load_pSCNN, \
                      predict_pSCNN, evaluate_pSCNN
from AutoRes.AutoRes import AutoRes, output_msp
from AutoRes.NetCDF import netcdf_reader  

if __name__=="__main__":
    # spectra = get_spectra_sqlite('dataset/NIST_Spec.db')
    # mz_ranges = get_mz_ranges(spectra)
    # plot_mz_hist(mz_ranges)
    # mz_range = (1, 1000)
    # spectra_filtered = filter_spectra(spectra, mz_range)
    # rand_sub_sqlite1(spectra_filtered, 'dataset/NIST_Spec-10000.db', 236283, 246283)
    # rand_sub_sqlite1(spectra_filtered, 'dataset/NIST_Spec0-236200.db', 0, 236200)
    '''
    c = sims('dataset/NIST_Spec0-236200.db', mz_range)
    with open('dataset/data.pk','wb') as file:
         pickle.dump(c, file)
    c1 = sims('dataset/NIST_Spec-10000.db', mz_range)
    with open('dataset/data1.pk','wb') as file:
         pickle.dump(c1, file)
    '''
    model_name1 = 'model/pSCNN1'
    model_name2 = 'model/pSCNN2'
    maxn1 = 1
    maxn2 = 3
    mz_range = (1, 1000)
    #train pSCNN1 model
    if check_pSCNN(model_name1):
        model1 = load_pSCNN(model_name1)
    else:
        para = {'dbname': 'dataset/NIST_Spec0-236200.db',
                'mz_range': mz_range, 
                'aug_num': 200000,
                'noise_level': 0.001,
                'maxn': maxn1,
                'layer_num': 3,
                'batch': 200,
                'epoch': 200,
                'lr': 0.001,
                'factor':0.8,
                'min_lr':0.000002,
                'model_name': model_name1}
        model1 = build_pSCNN(para)
    plot_loss_accuracy(model1)
    #test pSCNN1 model
    dbname1 = 'dataset/NIST_Spec-10000.db'
    spectra1 = get_spectra_sqlite(dbname1)
    convert_to_dense(spectra1, mz_range)  
    aug_eval1 = data_augmentation_1(spectra1, 100000, maxn1, 0.001)
    eval_acc1 = evaluate_pSCNN(model1, [aug_eval1['R'], aug_eval1['S']], aug_eval1['y'])
    yp1 = predict_pSCNN(model1, [aug_eval1['R'], aug_eval1['S']])    
    #train pSCNN2 model
    with open('dataset/data.pk', 'rb') as file_1:
         c = pickle.load(file_1)  
    if check_pSCNN(model_name2):
        model2 = load_pSCNN(model_name2)
    else:
        para = {'dbname': 'dataset/NIST_Spec0-236200.db',
                'mz_range': mz_range,
                'aug_num0': 100000,
                'aug_num1': 100000,
                'noise_level': 0.001,
                'maxn': maxn2,
                'layer_num': 2,
                'batch': 200,
                'epoch': 200,
                'lr': 0.0001,
                'factor':0.8,
                'min_lr':0.0000001,
                'c':c,
                'model_name': model_name2}
        model2 = build_pSCNN(para)
    plot_loss_accuracy(model2)
    #test pSCNN2 model
    with open('dataset/data1.pk', 'rb') as file_1:
         c1 = pickle.load(file_1) 
    spectra2 = get_spectra_sqlite('dataset/NIST_Spec-10000.db')
    convert_to_dense(spectra2, mz_range)
    aug_eval2 = data_augmentation_2(spectra2, c1, 10000, 90000, maxn2, 0.001)
    eval_acc2 = evaluate_pSCNN(model2, [aug_eval2['R'], aug_eval2['S']], aug_eval2['y'])
    yp2 = predict_pSCNN(model2, [aug_eval2['R'], aug_eval2['S']])

    #test AutoRes      
    model_names = ['model/pSCNN1','model/pSCNN2']   
    models = []
    for i in range(len(model_names)):
        model_name = model_names[i]
        model = load_pSCNN(model_name)
        models.append(model)
    model1 = models[0]
    model2 = models[1]
    plot_loss_accuracy(model1)
    plot_loss_accuracy(model2)
    mz_range = (1, 1000)
    filename = 'D:/CDF/06-1.0-3.CDF'
    sta_S0, area0, rt0, r_2_0 = AutoRes(filename, model1, model2)
    filename = 'msp/S-06.MSP'
    output_msp(filename, sta_S0, rt0)    
        
       

        
    
    