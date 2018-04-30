#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:18:00 2018
generate opensmile emotion based features on utterance level
@author: ddeng
"""
"""
///////// > openSMILE configuration file for emotion features <      //////////////////
/////////   Based on INTERSPEECH 2010 paralinguistics challenge      //////////////////
/////////   Pitch, Loudness, Jitter, MFCC, MFB, LSP and functionals  //////////////////
/////////                                                            //////////////////
/////////   1582 1st level functionals:                              //////////////////
/////////     (34 LLD + 34 delta) * 21 functionals                   //////////////////
/////////     +(4 LLD + 4 delta) * 19 functionals                    //////////////////
/////////     + 1 x Num. pitch onsets (pseudo syllables)             //////////////////
/////////      + 1 x turn duration in seconds                        //////////////////
"""

import csv
import glob
import os
import os.path
import subprocess
import tqdm
import numpy as np
from tqdm import tqdm
import pickle
from subprocess import call
import pdb
def clean_data(data):
    data = data.split(',')[1:] #'noname' deleted
    data[-1] = data[-1][0:-1] #'\n' deleted
    length = len(data)
    new_data = np.zeros(length)
    for i, item in enumerate(data):
        new_data[i] = float(item)
    return new_data[0:1582]

def save_as_pkl(folders, save_file_name):
    """
    main_Dic: {State_name:{Video_Name: {Utterance_index:[ feature_vector ]}}}

    """
    main_Dir = 'Audio_Feature_uttr_level'
    main_Dic = {}
    state_path = []
    for folder in folders:
        state_path.append( os.path.join(main_Dir, folder))
    for state in state_path:
        state_name = state.split('/')[-1]
        main_Dic[state_name] ={}
        video_path = glob.glob(os.path.join(state, '*'))
        for video in tqdm(video_path):
            pass
            video_name = video.split('/')[-1]
            main_Dic[state_name][video_name]={}

            utterance_path = glob.glob(os.path.join(video, '*.txt'))
            for utterance in utterance_path:
                utterance_index = utterance.split('/')[-1].split('.')[0].split('_')[-1]
                main_Dic[state_name][video_name][utterance_index]={}

                file = open(utterance, 'r')
                while True:
                    line = file.readline()
                    if line:
                        if line.startswith('@data'):
                            line = file.readline()
                            line = file.readline()
                            data = line
                            if data: #sometimes , the data might be empty
                                data = clean_data(data)
                                main_Dic[state_name][video_name][utterance_index]=data
                            break
                    else:
                        break

    with open(save_file_name ,'wb') as fout:
        pickle.dump(main_Dic, fout)

def check_already_extracted(feature_dir):
    """Check to see if we created the -0001 frame of this file."""

    return bool(os.path.exists(os.path.join(feature_dir ,'*.txt')))



def generate_aud_rep(folders):
    """
    extract 1582 features using OpenSmile Toolkit, the command line should be:
    ./SMILExtract -C ./config/emobase2010.conf -I /newdisk/test/000046280.wav -O /newdisk/test/000046280.txt
    -C : the configure file's path
    -I: the input .wav file
    -O: the output txt file
    Noting that SMILExtract executable file is in /openSMILE-2.1.0 in the home directory
    """
    opensmile_script_path = '/home/ddeng/openSMILE-2.1.0/SMILExtract'
    opensmile_conf = '/home/ddeng/openSMILE-2.1.0/config/emobase2010.conf'
    
    for folder in folders:
        video_folders = glob.glob(os.path.join('Audio_Frames', folder, '*'))
        print(folder + " videos extraction ...\n")

        for vid in tqdm(video_folders):
            print ("Video Path: "+ vid+'\n')
            Train_Valid_orTest = vid.split('/')[1]
            vid_name = vid.split('/')[-1]

    
            original_uttr_wav = glob.glob(os.path.join('../Audio_Frames', folder, vid_name, '*.wav'))
            for utter_wav in original_uttr_wav:
                utterance_name = utter_wav.split('/')[-1].split('.')[0]
                feature_path = os.path.join('Audio_Feature_uttr_level',Train_Valid_orTest, vid_name,utterance_name+'.txt')
                if not os.path.exists('Audio_Feature_uttr_level'):
                    os.mkdir('Audio_Feature_uttr_level')
                if not os.path.exists(os.path.join('Audio_Feature_uttr_level', Train_Valid_orTest)):
                    os.mkdir(os.path.join('Audio_Feature_uttr_level', Train_Valid_orTest))
                if not os.path.exists(os.path.join('Audio_Feature_uttr_level',Train_Valid_orTest,vid_name)):
                    os.mkdir(os.path.join('Audio_Feature_uttr_level',Train_Valid_orTest,vid_name))
 
                if not os.path.exists(feature_path):
                    des = feature_path
                    cmd = opensmile_script_path + ' -C '+opensmile_conf+' -I '+ utter_wav +' -O '+ des
                    process = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE, stderr= subprocess.STDOUT)
                    process.communicate()

                else:
                    print(vid + " feature has been extracted already.\n")




def main():
    """
    folders = ['Train', 'Validation']
    save_file_name =  'Audio_Feature_uttr_level.pkl' 
    generate_aud_rep(folders)
    save_as_pkl(folders, save_file_name)
    """
    folders = ['Test']
    save_file_name =  'Audio_Feature_uttr_level_Test.pkl'
    generate_aud_rep(folders)
    save_as_pkl(folders, save_file_name)

if __name__ == '__main__':
    pdb.set_trace()
    main()

