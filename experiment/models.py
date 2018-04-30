#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:56:01 2018

@author: ddeng
"""

from keras.layers import  Dense, Dropout, concatenate,Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model,Sequential, load_model
from keras.optimizers import Adam, SGD
from keras.layers.pooling import  AveragePooling1D, GlobalAveragePooling1D
from keras import metrics
from batch_renorm import BatchRenormalization
import sys
import functions

a, b=4.0, 8.0
class ResearchModels():
    def __init__(self, istrain= True, model='visual_model', seq_length = 20,
                 saved_model_path=None, task_type = 'arousal', 
                  saved_audio_model = None, saved_visual_model = None, 
                  saved_word_model = None,  learning_r = 1e-3):
        # set defaults
        self.istrain = istrain
        self.model = model
        self.seq_length = seq_length
        self.saved_model_path = saved_model_path
        self.task_type = task_type
        self.saved_audio_model = saved_audio_model
        self.saved_visual_model = saved_visual_model
        self.saved_word_model = saved_word_model
        # Get the appropriate model.
        if (self.saved_model_path is not None) :
            print("Loading model %s" % self.saved_model_path.split('/')[-1])
            self.model = self.load_custom_model(self.saved_model_path)
        elif model == 'visual_model':
            print("Loading visual model.")
            self.input_shape = (seq_length, 4096)
            self.model = self.visual_model()
        elif model == 'audio_model':
            print("Loading audio model.")
            self.input_shape = (1582,)
            self.model = self.audio_model()
        elif model == 'word_model':
            print("Loading word model.")
            self.input_shape = (10,)
            self.model = self.word_model()
        elif model == 'trimodal_model':
            print("Loading trimodal model")
            #self.model = self.trimodal_model_late_fusion()
            self.model = self.trimodal_model_early_fusion()
        elif model == 'bimodal_model':
            print("Loading bimodal model: audio and visual.")
            self.model = self.bimodal_model_audio_visual()
        else:
            print("Unknown network.")
            sys.exit()
        
        # Now compile the network.
        print (self.model.summary())
        sgd = SGD(lr = learning_r, decay = 1e-3, momentum = 0.9, nesterov = True)
        adam = Adam(lr = learning_r, decay = 1e-3, beta_1=0.9, beta_2=0.999)
        #adagrad = 
        self.model.compile(loss = 'mean_squared_error', metrics = [metrics.mse,functions.ccc_metric], optimizer = sgd)
        #self.model.compile(loss = functions.ccc_loss, metrics = [metrics.mse,functions.ccc_metric], optimizer = sgd)
        
    def load_custom_model(self,pretrained_model_path):
        model = load_model(pretrained_model_path, 
                           custom_objects ={'ccc_metric':functions.ccc_metric,
                                            'ccc_loss': functions.ccc_loss})
        return model
        
    def decision_layer(self,name):
        name0 = name
        if self.task_type == 'arousal':
            # add the output layer
            dl=Dense(1, activation ='sigmoid' , kernel_initializer ='normal', name = name0+'_decision_layer')
            
        elif self.task_type == 'valence':
            dl = Dense(1, activation= 'tanh' ,kernel_initializer ='normal',name = name0+'_decision_layer' )
            
        elif self.task_type == 'emotion':
            dl = Dense(7, activation= 'softmax' ,kernel_initializer ='normal',name = name0+'_decision_layer')
            
        return dl
    
    def add_hidden_layer(self, model, name):
        name0 = name
        # the hidden layer block
        model.add(Dense(256, name = name0+'_hidden_layer'))
        model.add(Activation('relu', name = name0+'_activation'))
        model.add(BatchNormalization(name = name0+'_BN'))
        model.add(Dropout(0.5, name = name0+'_dropout'))

        return model

    def visual_model(self):
        # when input is visual feature
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.seq_length,4805), name = 'visual_BN_1'))
        model.add(AveragePooling1D(pool_size = 2 , name ='visual_average'))
        
        
        # lstm layer
        model.add(LSTM(64, name  = 'visual_lstm'))
        model.add(Activation('relu', name = 'visual_activation1'))
        model.add(BatchNormalization(name ='visual_BN_2'))
        model.add(Dropout(0.5, name = 'visual_dropout_1'))
        
        # the hidden layer
        model = self.add_hidden_layer(model, 'visual')
        
        # the decision layer
        model.add(self.decision_layer('visual'))
        
        return model
        
    def word_model(self):
        model = Sequential()
        model.add(BatchNormalization(input_shape = (10,), name = 'word_BN_1'))
        # add the hidden layer
        model = self.add_hidden_layer(model,'word')

        #add the decision layer
        model.add(self.decision_layer('word'))
        return  model
    

    def audio_model(self):
        model = Sequential()
        # the input layer
        model.add(BatchNormalization(input_shape = (1582,), name = 'audio_BN_1'))
   
        # add the hidden layer
        model = self.add_hidden_layer(model,'audio')

        #add the decision layer
        model.add(self.decision_layer('audio'))
        return  model
       
       
    def trimodal_model_early_fusion(self):
        
        #audio model
        if self.saved_audio_model:
            audio_model = self.load_custom_model(self.saved_audio_model)
        else:
            audio_model = self.audio_model()
        #visual model
        if self.saved_visual_model:
            visual_model = self.load_custom_model(self.saved_visual_model)
        else:
            visual_model = self.visual_model()
        # word model
        if self.saved_word_model:
            word_model = self.load_custom_model(self.saved_word_model)
        else:
            word_model = self.word_model()
            
        #get rid of decision layers
        audio_model.layers.pop()
        visual_model.layers.pop()
        word_model.layers.pop()
        
        #input and output
        audio_input = audio_model.input
        audio_output = audio_model.layers[-1].output
        visual_input = visual_model.input
        visual_output = visual_model.layers[-1].output
        word_input = word_model.input
        word_output = word_model.layers[-1].output
        
        concat_layer = concatenate([audio_output, visual_output, word_output])
    
        x = Dense(1024)(concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        out = self.decision_layer('trimodal')(x)
        
        fusion_model = Model([audio_input, visual_input, word_input], out)
        
        return fusion_model
    def trimodal_model_late_fusion(self):
        #audio model
        if self.saved_audio_model:
            audio_model = self.load_custom_model(self.saved_audio_model)
        else:
            audio_model = self.audio_model()
        #visual model
        if self.saved_visual_model:
            visual_model = self.load_custom_model(self.saved_visual_model)
        else:
            visual_model = self.visual_model()
        # word model
        if self.saved_word_model:
            word_model = self.load_custom_model(self.saved_word_model)
        else:
            word_model = self.word_model()
            #input and output
        audio_input = audio_model.input
        audio_output = audio_model.layers[-1].output
        visual_input = visual_model.input
        visual_output = visual_model.layers[-1].output
        word_input = word_model.input
        word_output = word_model.layers[-1].output
        
        merge_layer = concatenate([audio_output, visual_output, word_output])
        out = Dense(1)(merge_layer)
        
        fusion_model = Model([audio_input, visual_input, word_input], out)
        
        return fusion_model
    def bimodal_model_audio_visual(self):
        # audio model
        audio_model = Sequential()
        audio_model.add(BatchNormalization(input_shape = (1582,), name = 'av_audio_BN_1'))
        audio_model = self.add_hidden_layer(audio_model,'av_audio')
        
        #visual model
        visual_model = Sequential()
        visual_model.add(BatchNormalization(input_shape = (self.seq_length,4805), name = 'av_visual_BN_1'))
        visual_model.add(AveragePooling1D(pool_size = 2 , name ='av_visual_average'))
        visual_model.add(LSTM(64, name  = 'av_visual_lstm'))
        visual_model.add(Activation('relu', name = 'av_visual_activation1'))
        visual_model.add(BatchNormalization(name ='av_visual_BN_2'))
        visual_model.add(Dropout(0.5, name = 'av_visual_dropout_1'))
        visual_model = self.add_hidden_layer(visual_model, 'av_visual')
        
        audio_input = audio_model.input
        audio_output = audio_model.layers[-1].output
        visual_input = visual_model.input
        visual_output = visual_model.layers[-1].output
        
        concat_layer = concatenate([audio_output, visual_output])
        
        x  = Dense(1024) (concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        out = self.decision_layer('bimodal-av')(x)
        
        fusion_model = Model([audio_input, visual_input], out)
        return fusion_model
    
    def bimodal_model_audio_word(self):
        # audio model
        audio_model = Sequential()
        audio_model.add(BatchNormalization(input_shape = (1582,), name = 'aw_audio_BN_1'))
        audio_model = self.add_hidden_layer(audio_model,'aw_audio')
        
        #word model
        word_model = Sequential()
        word_model.add(BatchNormalization(input_shape = (10,), name = 'aw_word_BN_1'))
        # add the hidden layer
        word_model = self.add_hidden_layer( word_model,'aw_word')
        
        audio_input = audio_model.input
        audio_output = audio_model.layers[-1].output
        word_input = word_model.input
        word_output = word_model.layers[-1].output
        
        concat_layer = concatenate([audio_output, word_output])
        
        x  = Dense(1024) (concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        out = self.decision_layer('bimodal-aw')(x)
        
        fusion_model = Model([audio_input, word_input], out)
        return fusion_model
    
    def bimodal_model_visual_word(self):
        #visual model
        visual_model = Sequential()
        visual_model.add(BatchNormalization(input_shape = (self.seq_length,4805), name = 'vw_visual_BN_1'))
        visual_model.add(AveragePooling1D(pool_size = 2 , name ='vw_visual_average'))
        visual_model.add(LSTM(64, name  = 'vw_visual_lstm'))
        visual_model.add(Activation('relu', name = 'vw_visual_activation1'))
        visual_model.add(BatchNormalization(name ='vw_visual_BN_2'))
        visual_model.add(Dropout(0.5, name = 'vw_visual_dropout_1'))
        visual_model = self.add_hidden_layer(visual_model, 'vw_visual')
        #visual model
        word_model = Sequential()
        word_model.add(BatchNormalization(input_shape = (10,), name = 'vw_word_BN_1'))
        # add the hidden layer
        word_model = self.add_hidden_layer( word_model,'vw_word')
         
        visual_input = visual_model.input
        visual_output = visual_model.layers[-1].output
        word_input = word_model.input
        word_output = word_model.layers[-1].output
        
        concat_layer = concatenate([visual_output, word_output])
        
        x  = Dense(1024) (concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        out = self.decision_layer('bimodal-vw')(x)
        
        fusion_model = Model([visual_input, word_input], out)
        return fusion_model

