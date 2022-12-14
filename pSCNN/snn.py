from tensorflow.keras import Input, layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle, os
from pSCNN.db import get_spectra_sqlite, convert_to_dense
from pSCNN.da import data_augmentation_1, data_augmentation_2
import matplotlib.ticker as mtick

def create_input_layers(xshapes):
    inputs = []
    for xshape in xshapes:
        input_shape_x = (xshape[1], 1)
        input_x = Input(shape = input_shape_x)
        inputs.append(input_x)
    return inputs

def create_convolution_layers(inputs, num_layers = 0):
    convs = []
    for input_x in inputs:
        conv      = layers.Conv1D(32,  5,  activation='relu', input_shape=input_x.get_shape())(input_x)
        conv      = layers.MaxPooling1D(strides=2, padding='valid')(conv)
        conv      = layers.Dropout(0.1)(conv)
        for i in range(num_layers):
            conv      = layers.Conv1D(32*(i+2),  5,  activation='relu')(conv)
            conv      = layers.MaxPooling1D(strides=2, padding='valid')(conv)
            conv      = layers.Dropout(0.1)(conv)
        convs.append(conv)
    return convs

def pSCNN(xshapes, lr, num_conv_layers):
    inputs = create_input_layers(xshapes)
    convs = create_convolution_layers(inputs, num_layers = num_conv_layers)
    if len(convs) >= 2:
        conv_merge = layers.concatenate(convs)
    else:
        conv_merge = convs[0]
    flat      = layers.Flatten()(conv_merge)
    dense     = layers.Dense(100,  activation='relu')(flat)
    dense     = layers.Dropout(0.21)(dense)
    output    = layers.Dense(1, activation='sigmoid')(dense)
    model     = models.Model(inputs= inputs, outputs=output)
    model.compile(optimizer=optimizers.Adam(lr=lr), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_pSCNN(model, Xs, y, fa, batch, epochs, min_lr):
    Xs3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    callbacks = [EarlyStopping(patience=60, verbose=1), 
                 ReduceLROnPlateau(factor=fa, patience=1, min_lr=min_lr, verbose=1),
                 ModelCheckpoint('D:/MODEL.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)]
    model.fit(Xs3d, y, batch_size=batch, epochs=epochs, callbacks=callbacks, validation_split = 0.1)
        
def plot_loss_accuracy(model):
    history = model.history.history
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    plot1 = ax1.plot(history['accuracy'], label = 'Acc_training')
    plot2 = ax1.plot(history['val_accuracy'], label = 'Acc_validation')
    ax2 = ax1.twinx()
    plot3 = ax2.plot(history['loss'], label = 'Loss_training', color='#2ca02c')
    plot4 = ax2.plot(history['val_loss'], label = 'Loss_validation', color='#d62728')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    lines = plot1 + plot2 + plot3 + plot4
    ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(0.98, 0.95))

def build_pSCNN(para):
    '''
    Build and train the model
    
    input
        para: Hyperparameters for model training
    
    output
        model
    '''
    spectra = get_spectra_sqlite(para['dbname'])
    convert_to_dense(spectra, para['mz_range'])
    if para['maxn'] == 1:
        aug = data_augmentation_1(spectra, para['aug_num'], para['maxn'], para['noise_level'])
    else:
        aug = data_augmentation_2(spectra, para['c'], para['aug_num0'], para['aug_num1'], para['maxn'], para['noise_level'])
    model = pSCNN([aug['R'].shape, aug['S'].shape], para['lr'], para['layer_num'])
    train_pSCNN(model, [aug['R'], aug['S']], aug['y'], para['factor'], para['batch'], para['epoch'], para['min_lr'])
    save_pSCNN(model, para['model_name'])
    return model

def save_pSCNN(model, model_name):
    model_path = f'{model_name}.h5'
    history_path = f'{model_name}.pkl'
    model.save(model_path)
    pickle.dump(model.history.history, open(history_path, "wb" ))

def load_pSCNN(model_name):
    '''
    Load the trained model
    
    input
        model_name: model name
    
    output
        model
    '''
    model_path = f'{model_name}.h5'
    history_path = f'{model_name}.pkl'
    model = models.load_model(model_path) 
    history = pickle.load(open(history_path, "rb" ))
    model.history = callbacks.History()
    model.history.history = history
    return model 

def check_pSCNN(model_name):
    model_path = f'{model_name}.h5'
    history_path = f'{model_name}.pkl'
    return os.path.isfile(model_path) and os.path.isfile(history_path) 

def predict_pSCNN(model, Xs):
    '''
    Model Prediction
    
    input
        model: pSCNN
        Xs： Model Input Data
    
    output
        Model prediction results
    '''
    Xs3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    return model.predict(Xs3d)

def evaluate_pSCNN(model, Xs, y):
    '''
    Model Evaluation
    
    input
        model: pSCNN
        Xs： Model Input Data
        y: 
    output
        Model prediction results
    '''
    Xs3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    return model.evaluate(Xs3d, y)
