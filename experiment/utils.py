#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:09:51 2018

@author: ddeng
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas
import sys
sys.path.append('..')
from calculateEvaluationCCC import ccc, mse
import pickle
def read_log(file_path):
    # read log and return dictionary
    # epoch,ccc_metric,loss,mean_squared_error,val_ccc_metric,val_loss,val_mean_squared_error
    data = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            if index ==0:
                keys = line.rstrip('\r\n').split(',')
                nums = len(keys)
                for key in keys:
                    data[key] = []
            else:
                numbers = line.rstrip('\r\n') .split(',')
                for i in range(nums):
                    data[keys[i]]  = float(numbers[i])
    return data
            
def display_true_vs_pred(y_true, y_pred, log_path, task, model):
    #display the true label vs prediction using image
    if not os.path.exists('images'):
        os.mkdir('images')
    name = log_path.split('/')[-1].split('.')[0]
    name = os.path.join('images', name)
    my_list = ['Validation','Train']
    # draw the y_true and y_pred plot
    for index,( y_t, y_p) in enumerate(zip(y_true, y_pred)):
        plt.figure()
        plt.scatter(y_p, y_t)
        plt.xlabel('Prediction in {} set'.format(my_list[index]))
        plt.ylabel('True label in {} set '.format(my_list[index]))
        if task == 'arousal':
            axes = plt.gca()
            axes.set_xlim([0,1])
            axes.set_ylim([0,1])
            plt.text(0.1, 0.9,'CCC: %.3f\nMSE: %.3f'%(ccc(y_t, y_p)[0],mse(y_t, y_p)),
         horizontalalignment='center',
         verticalalignment='center')
        elif task == 'valence':
            axes = plt.gca()
            axes.set_xlim([-1,1])
            axes.set_ylim([-1,1])
            plt.text(-0.8, 0.8,'CCC: %.3f\nMSE: %.3f'%(ccc(y_t, y_p)[0],mse(y_t, y_p)),
         horizontalalignment='center',
         verticalalignment='center')
        elif task == 'emotion':
            print("confusion matrix...")
        
        plt.savefig(name +'-'+model+'-%d-true_vs_pred.png'%index)
        plt.show()
    
def videoset_csv_reader(csv_file, dictionary, istrain):
    # read omg_TrainVideos.csv and omg_ValidationVideos.csv
    counter = 0
    with open(csv_file, 'r') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            video = row['video']
            utterance = row['utterance']
            utter_index = utterance.split('.')[0].split('_')[-1]
            if istrain:
                m_labels = [float(row['arousal']), float(row['valence']), int(row['EmotionMaxVote'])]
            else:
                m_labels = []
            if video not in dictionary.keys():
                dictionary[video] = {}
            dictionary[video][utter_index] = m_labels
            counter +=1
    return counter

def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        return pickle.load(fin)
    
def print_out_csv(arousal_pred, valence_pred, name_list, refer_csv, out_csv):
    data = {}
    #laod prediction
    for index, ids in enumerate(name_list):
        vid,uttr = ids
        prediction = [arousal_pred[index], valence_pred[index]]
        if vid not in data.keys():
            data[vid] = {}
        data[vid][uttr] = prediction
    # print out in order
    df = pandas.read_csv(refer_csv)
    videos = df['video']
    utterances = df['utterance']
    new_df = pandas.DataFrame({'video':[], 'utterance':[], 'arousal':[],'valence':[]})
    new_df['video'] = videos
    new_df['utterance'] = utterances
    arousal = []
    valence = []
    for vid,utter in zip(videos, utterances):
        uttr_index = utter.split('.')[0].split('_')[-1]
        try:
            a = data[vid][uttr_index][0]
        except:
            print("{} arousal prediction is missing!".format(vid+':'+utter))
            a = 0.0
        try:
            v = data[vid][uttr_index][1]
        except:
            print("{} arousal prediction is missing!".format(vid+':'+utter))
            v = 0.0
        arousal.append(a)
        valence.append(v)
    new_df['arousal'] = arousal
    new_df['valence'] = valence
    new_df[['video', 'utterance','arousal','valence']].to_csv(out_csv, index=False)
    print("csv file printed out successfully!")
            
