#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:00:38 2018

@author: ddeng
"""
import pickle
import pdb
def read_file(file):
    main_dict = {}
    for line in file:
        line = line.strip('\n')
        line = line.strip('\t')
        [vid, uttr, pos_uttr, neg_uttr, pos_vid, neg_vid,words_uttr] = line.split(' ')    
        if vid not in main_dict.keys():
            main_dict[vid] = {}
        uttr_index = uttr.split('.')[0].split('_')[-1]
        main_dict[vid][uttr_index] = [float(pos_uttr), float(neg_uttr), float(pos_vid), float(neg_vid)]
    return main_dict
def parse_MPQA_feature():
    train_feature_path = 'omg_TrainTranscripts_features.txt'
    valid_feature_path = 'omg_ValidationTranscripts_features.txt'
    data = {}
    train_file = open (train_feature_path,'r')
    data['Train'] = read_file(train_file)
    valid_file = open(valid_feature_path, 'r')
    data['Validation'] = read_file(valid_file)
    
    with open('../MPQA_Word_Feature.pkl','wb') as fout:
        pickle.dump(data, fout)
def parse_MPQA_feature_for_test():
    test_feature_path = 'omg_TestTranscripts_features.txt'
    data = {}

    test_file = open(test_feature_path,'r')
    data['Test'] = read_file(test_file)
    with open('../MPQA_Word_Feature_Test.pkl','wb') as fout:
        pickle.dump(data, fout)
if __name__ == '__main__':
    pdb.set_trace()
    #parse_MPQA_feature()
    parse_MPQA_feature_for_test()
