#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:51:38 2018
run it on python3 instead of python2
@author: ddeng
"""
from string import punctuation
import nltk
from nltk.corpus import opinion_lexicon
from collections import Counter
from nltk.corpus import stopwords
import csv
import pdb
import pickle
from tqdm import tqdm
#nltk.download('opinion_lexicon')
def get_ids(row):
    if row:
        uttr_index = row[-2].split('.')[0].split('_')[-1]
        vid = row[-3]
        return [vid, uttr_index]
    else:
        return []

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

def process_docs(csv_file):
	# read csv_file, and go through all utterances
    # count the number of positive and negtive words
    word_feature = {}
    waiting_list = []
    with open(csv_file, 'r') as file:
        filereader = csv.reader(file)
        # skip the first row
        index =0
        for row in tqdm(filereader):
            pass
            if index != 0:
                counter_uttr_pos = 0
                counter_uttr_neg = 0
                counter_all_words = 0
                uttr_tokens = clean_doc(row[-1])
                vid, uttr = get_ids(row)
                if vid not in word_feature.keys():
                    word_feature[vid] = {}
                if len(uttr_tokens) == 0:
                # if empty, take the previous transcript if exists
                # else, take the nearest following ones
                    if len(word_feature[vid].keys()) != 0:
                        last_uttr = sorted(word_feature.keys())[-1]
                        word_feature[vid][uttr] = word_feature[vid][last_uttr]
                    else:
                        waiting_list.append([vid,uttr])
                    break    
                for token in uttr_tokens:
                    if token in opinion_lexicon.positive():
                        counter_uttr_pos+=1
                    elif token in opinion_lexicon.negative():
                        counter_uttr_neg+=1
                    counter_all_words+= 1
                word_feature[vid][uttr] = [counter_uttr_pos, counter_uttr_neg, counter_all_words, 0, 0, 0]
                for vid0, uttr0 in waiting_list:
                    word_feature[vid0][uttr0] = word_feature[vid][uttr]
                waiting_list = []   
            index+=1
        for vid in word_feature.keys():
            pass
            counter_vid_pos = sum([word_feature[vid][uttr][0] for uttr in word_feature[vid].keys()])
            counter_vid_neg = sum([word_feature[vid][uttr][1] for uttr in word_feature[vid].keys()])
            counter_vid_all_words = sum([word_feature[vid][uttr][2] for uttr in word_feature[vid].keys()])
            for uttr in word_feature[vid].keys():
                word_feature[vid][uttr][3] = counter_vid_pos
                word_feature[vid][uttr][4] = counter_vid_neg
                word_feature[vid][uttr][5] = counter_vid_all_words 
    return word_feature
def save_as_pkl(obj, file_name):
     with open(file_name,'wb') as fout:
        pickle.dump(obj, fout, 2)


pdb.set_trace()
"""
word_feature = {}
word_feature['Train'] = process_docs('../omg_TrainTranscripts.csv')
word_feature['Validation'] = process_docs('../omg_ValidationTranscripts.csv')

save_as_pkl(word_feature,'Word_Feature.pkl')
"""
word_feature = {}
word_feature['Test'] = process_docs('../omg_TestTranscripts.csv')
save_as_pkl(word_feature,'Word_Feature_Test.pkl')   
