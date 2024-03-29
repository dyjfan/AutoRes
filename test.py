import pickle, os, sys, time
from pSCNN.db import get_spectra_sqlite, get_mz_ranges, rand_sub_sqlite1, \
                      convert_to_dense, plot_mz_hist, filter_spectra
from pSCNN.da import data_augmentation_1, data_augmentation_2
from pSCNN.snn import plot_loss_accuracy, check_pSCNN, build_pSCNN, load_pSCNN, \
                      predict_pSCNN, evaluate_pSCNN
from AutoRes.AutoRes import AutoRes, output_msp, AutoRes_alignment
import pandas as pd
from AutoRes.NetCDF import netcdf_reader  
import warnings
import wget
from zipfile import ZipFile

if __name__=="__main__":
    '''
    # Splitting the dataset
    # It should be noted that the NIST_Spec.db database is not available for download due to copyright reasons.
    spectra = get_spectra_sqlite('dataset/NIST_Spec.db')
    mz_ranges = get_mz_ranges(spectra)
    plot_mz_hist(mz_ranges)
    mz_range = (1, 1000)
    spectra_filtered = filter_spectra(spectra, mz_range)
    rand_sub_sqlite1(spectra_filtered, 'dataset/test_sublibrary.db', 257376, 267376)
    rand_sub_sqlite1(spectra_filtered, 'dataset/training_sublibrary.db', 0, 257376)
    c = sims('dataset/training_sublibrary.db', mz_range)
    with open('dataset/data.pk','wb') as file:
         pickle.dump(c, file)
    c1 = sims('dataset/test_sublibrary.db', mz_range)
    with open('dataset/data1.pk','wb') as file:
         pickle.dump(c1, file)
    '''
    # Download plant essential oil dataset
    url = "https://github.com/dyjfan/AutoRes/releases/download/1.0/data.zip"
    save_path = wget.download(url, ".")
    with ZipFile(save_path, 'r') as zip:
        zip.extractall(path = ".")
    path = save_path[:-4]
    
    mz_range = (1, 1000)
    model_name1 = 'model/pSCNN1'
    model_name2 = 'model/pSCNN2'
    maxn1 = 1
    maxn2 = 3
    
    # train pSCNN1 model
    if check_pSCNN(model_name1):
        model1 = load_pSCNN(model_name1)
    else:
        para = {'dbname': 'dataset/training_sublibrary.db',
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
    # test pSCNN1 model
    dbname1 = 'dataset/test_sublibrary.db'
    spectra1 = get_spectra_sqlite(dbname1)
    convert_to_dense(spectra1, mz_range)  
    aug_eval1 = data_augmentation_1(spectra1, 100000, maxn1, 0.001)
    eval_acc1 = evaluate_pSCNN(model1, [aug_eval1['R'], aug_eval1['S']], aug_eval1['y'])
    yp1 = predict_pSCNN(model1, [aug_eval1['R'], aug_eval1['S']])    
    
    # train pSCNN2 model
    with open('dataset/data.pk', 'rb') as file_1:
         c = pickle.load(file_1)  
    if check_pSCNN(model_name2):
        model2 = load_pSCNN(model_name2)
    else:
        para = {'dbname': 'dataset/training_sublibrary.db',
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
    # test pSCNN2 model
    with open('dataset/data1.pk', 'rb') as file_1:
         c1 = pickle.load(file_1) 
    spectra2 = get_spectra_sqlite('dataset/test_sublibrary.db')
    convert_to_dense(spectra2, mz_range)
    aug_eval2 = data_augmentation_2(spectra2, c1, 10000, 90000, maxn2, 0.001)
    eval_acc2 = evaluate_pSCNN(model2, [aug_eval2['R'], aug_eval2['S']], aug_eval2['y'])
    yp2 = predict_pSCNN(model2, [aug_eval2['R'], aug_eval2['S']])
    
    # test AutoRes
    start = time.time()
    files = os.listdir(path)
    Alignment_or_not = True
    if Alignment_or_not:
        df = AutoRes_alignment(path, files[0:5], model1, model2)
        df.to_csv(path + '/' + 'SetⅠ.csv', index=False)
        df = AutoRes_alignment(path, files[5:10], model1, model2)
        df.to_csv(path + '/' + 'SetⅡ.csv', index=False)
    else:
        for filename in files:
            ncr = netcdf_reader(path + '/' + filename, True)
            sta_S, area, rt, R2 = AutoRes(ncr, model1, model2, filename)
            msp = filename.split('.CDF')[0] + '.MSP'
            output_msp(path + '/'+ msp, sta_S, rt)
            csv = filename.split('.CDF')[0] + '.csv'
            df = pd.DataFrame({'rt': rt, 'area': area, 'R2': R2})
            df.to_csv(path + '/' + csv, index = False)
    end = time.time()
    print("Running time: %.2fs"%(end - start))
