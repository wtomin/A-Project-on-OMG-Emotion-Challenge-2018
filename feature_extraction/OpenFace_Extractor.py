#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:45:29 2018
extract features using OpenFace
@author: ddeng
"""
import os
import numpy as np
import glob
from subprocess import call
import subprocess
import pdb
from tqdm import tqdm
import csv
import pickle
import pandas as pd
Video_folder = '/newdisk/OMGEmotionChallenge/Videos'
OpenFace_Feature_folder = '/newdisk/OMGEmotionChallenge/OpenFace_Feature'
OpenFace_Extractor_path = '/home/ddeng/OpenFace/build/bin/FeatureExtraction'
seq_length = 20

def Feature_extractor(folders):
    
    
    for folder in folders:
            
        videos = glob.glob(os.path.join(Video_folder, folder, '*'))
        videos.remove(os.path.join(Video_folder, folder, 'youtube_videos_temp'))
        
        
        for video in tqdm(videos):
            pass
            video_name = video.split('/')[-1]
            des = os.path.join(OpenFace_Feature_folder, folder, video_name)
            #check existence
            if not os.path.exists(OpenFace_Feature_folder):
                os.mkdir(OpenFace_Feature_folder) 
            if not os.path.exists(os.path.join(OpenFace_Feature_folder, folder)):
                os.mkdir(os.path.join(OpenFace_Feature_folder, folder))
            if not os.path.exists(os.path.join(OpenFace_Feature_folder, folder, video_name)):
                os.mkdir(os.path.join(OpenFace_Feature_folder, folder, video_name))
            
            os.chdir(des)    
            utterance_videos = glob.glob(os.path.join(video, '*.mp4'))
            cmd = OpenFace_Extractor_path
            if not os.path.exists(os.path.join(des, 'processed')):
                for uttr in utterance_videos:
                    cmd = cmd + ' -f '+uttr
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr= subprocess.STDOUT,universal_newlines=True)
                process.communicate()
def read_csv_return_face_feature(file_path):
    """
    frame, face_id, timestamp, confidence, success, gaze_0_x, ...
    from an utternace frames, 
    """
    data = []
    df = pd.read_csv(file_path)
    confidence_index = [ i for i, s in enumerate(df[df.columns[4]]) if float(s) == 1]
    if len(confidence_index) == 0:
        # no face detected 
        return None
    length = len(confidence_index)
    taken_index = []
    if length<seq_length:
        strate = 'repeat_final'
        final_index = confidence_index[-1]
        taken_index = confidence_index
    else:
        strate = 'equal_interval'
        interval = length//seq_length 
        for i in range(seq_length):
            taken_index.append(confidence_index[i*interval])
    with open(file_path,'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for index, row in enumerate(reader):
            if index in taken_index:
                if (strate == 'repeat_final') and (index == final_index):
                    for i in range(seq_length - length +1 ):
                        data.append( float_a_list(row[5:]))
                else:
                    data.append(float_a_list(row[5:]))
        data = np.asarray(data)
        assert data.shape[0] == seq_length 
        return data
def get_keys(uttr_csv_file_path):
    # uttr_csv_file_path : /OpenFace_Feature_folder/Train/Video_name/processed/utterance_x.csv
    parts = uttr_csv_file_path.split('/')
    utterance_index = parts[-1].split('.')[0].split('_')[-1]
    video_name = parts[-3]
    state_name = parts[-4]
    
    return [state_name, video_name, utterance_index]

def float_a_list(list):
    new_list = []
    for item in list:
        new_list.append(float(item))
    return new_list
def save_as_dict(data, uttr_csv_file_path, dictionary):
    # data : 20 frame, feature
    # uttr_csv_file_path : /OpenFace_Feature_folder/Train/Video_name/processed/utterance_x.csv
    state_name, video_name, utterance_index = get_keys(uttr_csv_file_path)
    if not (state_name in dictionary.keys()):
        dictionary[state_name] = {}
    if not (video_name in dictionary[state_name].keys()):
        dictionary[state_name][video_name] = {}
    
    dictionary[state_name][video_name][utterance_index] = data
    
def save_feature(folders, feature_file_name):
    features = {}
    
    for folder in folders:
       videos = glob.glob(os.path.join(OpenFace_Feature_folder, folder, '*')) 
       
       for video in tqdm(videos):
           pass
           processed_video = os.path.join(video, 'processed')
           utter_csvs = glob.glob(os.path.join(processed_video, 'utterance_*.csv'))
           for utter_csv in utter_csvs:
               
               data_file = read_csv_return_face_feature(utter_csv)
               if data_file is not None:
                  save_as_dict(data_file, utter_csv, features)
               else:
                   print(utter_csv+"feature doesn't exist!")
    with open(feature_file_name, 'wb') as fout:
        pickle.dump(features, fout)
def main():
    #For train set and validation set:
    #run Feature_extractor() first, then run save_feature()
    #folders = ['Train', 'Validation']
    #feature_file_name = 'OpenFace_Feature.pkl'
    #Feature_extractor(folders)
    #save_feature(folders)


    # for the test set, run 
    folders = ['Test']
    feature_file_name = 'OpenFace_Feature_Test.pkl'
    Feature_extractor(folders)
    save_feature(folders, feature_file_name)
if __name__ == '__main__':
    pdb.set_trace()
    main()
           
