"""
Class for managing our data.
noting that the we use the following files:

omg_TrainVideos.csv : link,start, end, video, utterance, arousal, valence, and emotionvote
omg_Validation.csv:  link, start, end,video, utterance, arousal, valence, and emotionvote

Audio_Feature_utter_level.pkl: a dictionary, dict['Train']['video_name']['utterance_index']= audio_feature_vec (1582)
So it is with 'fc6-vgg16.pkl', 'OpenFace_Feature.pkl', 'Word_Feature.pkl'

"""

import numpy as np
import random
random.seed(123)
import os.path
import pickle
from keras.utils.np_utils import to_categorical
from utils import videoset_csv_reader, load_pickle
from sklearn.preprocessing import MinMaxScaler

class DataSet():

    def __init__(self, istrain=True, model = 'visual_model', task = 'arousal',seq_length=20):  # initialization, the length of sequences in one video is 20 (modifiable)
        self.istrain = istrain
        self.model = model
        self.task = task
        self.seq_length = seq_length
        self.audio_f_dim = 1582
        self.visual_f_dim = 4805
        self.word_f_dim = 10
        
        self.train_video_csv_path = os.path.join('..','omg_TrainVideos.csv')
        self.validation_video_csv_path = os.path.join('..','omg_ValidationVideos.csv')

        self.test_video_csv_path = os.path.join('..','omg_TestVideos.csv')

        #self.test_video_path = os.path.join('..','new_omg_ValidationVideos.csv')
        # Get the data, including video, utterance and labels
        self.data = self.get_data()
        # where the visual features are stored, there are two sources
        self.visual_feature_path0 = [os.path.join('..','feature_extraction','Visual_Feature','fc6-vgg16.pkl'),
                                     os.path.join('..','feature_extraction','Visual_Feature','Test-fc6-vgg16.pkl')]
        self.visual_feature_path1 = [os.path.join('..','feature_extraction','OpenFace_Feature.pkl'),
                                     os.path.join('..','feature_extraction','OpenFace_Feature_Test.pkl')]
        # where the audio_feature is stored
        self.audio_feature_path = [os.path.join('..','feature_extraction','Audio_Feature_uttr_level.pkl'),
                                   os.path.join('..','feature_extraction', 'Audio_Feature_uttr_level_Test.pkl')]
        
        #where the word feature is stored
        self.word_feature_path0 = [os.path.join('..','feature_extraction','Word_Feature.pkl'),
                                   os.path.join('..','feature_extraction', 'Word_Feature_Test.pkl')]
        self.word_feature_path1 = [os.path.join('..','feature_extraction','MPQA_Word_Feature.pkl'),
                                   os.path.join('..','feature_extraction', 'MPQA_Word_Feature_Test.pkl')]

        self.load_neccessary(self.model)

 
    def get_data(self):
        data = {}

        # train data stored in data['train']
        data['Train'] = {}
        train_counter = videoset_csv_reader(self.train_video_csv_path, data['Train'], self.istrain)
    
        # validation data stored in data['validation']
        data['Validation'] = {}
        valid_counter = videoset_csv_reader(self.validation_video_csv_path, data['Validation'], self.istrain)
        print("The dataset has been split to: train set:{} videos, {} utterances; validation set: {} videos, {} utterances. ".format(
            len(data['Train'].keys()), train_counter, 
            len(data['Validation'].keys()), valid_counter))
        if not self.istrain:
        # test data stored in data['validation']
            data['Test'] = {}
            test_counter = videoset_csv_reader(self.test_video_csv_path, data['Test'], self.istrain)
            print("Evaluation for test set: {} videos, {} utterances. ".format(
                   len(data['Test'].keys()), test_counter))

        return data

    def unroll_and_normalize(self, feature_dict):
        # So the normalization will be done in train set, validation set and (test set)
        normalized_f_dict = {}
        scaler = MinMaxScaler()
        if self.istrain:
            states = ['Train', 'Validation']
        else:
            states = ['Train', 'Validation','Test']
        for state in states:
            main_dict = feature_dict[state]
            videos = main_dict.keys()
            features = []
            indexes = []
            
            for vid in videos:
                utterances = main_dict[vid].keys()
                for uttr in utterances:
                    indexes.append([vid, uttr])
                    feature = np.asarray(main_dict[vid][uttr])
                    if len(feature.shape) == 2:
                        #(time_steps, feature_dim)
                        time_steps = feature.shape[0]
                        feature_dim = feature.shape[1]
                        is_time = True
                    elif len(feature.shape) == 1:
                        feature_dim = feature.shape[0]
                        is_time = False
                    features.append(feature)
            #reshape if time dimension exists, and normalize by scale
            if is_time:
                unrolled_f = np.asarray(features).reshape((-1,feature_dim))
                if state == 'Train':
                    scaled_f = scaler.fit_transform(unrolled_f)
                    scaled_f = scaled_f.reshape((-1, time_steps, feature_dim))
                else:
                    scaled_f = scaler.transform(unrolled_f)
                    scaled_f = scaled_f.reshape((-1, time_steps, feature_dim))
            else:
                unrolled_f  = np.asarray(features)
                if state == 'Train':
                    scaled_f = scaler.fit_transform(unrolled_f)
                else:
                    scaled_f = scaler.transform(unrolled_f)
            for i in range(scaled_f.shape[0]):
                vid,uttr = indexes[i]
                main_dict[vid][uttr] = scaled_f[i]
            normalized_f_dict[state] = main_dict
        return normalized_f_dict
                    
    def load_neccessary(self, model_type):
        if model_type == 'trimodal_model':
            self.visual_feature = self.unroll_and_normalize(self.load_fused_visual_feature())
            self.audio_feature = self.unroll_and_normalize(self.load_feature(self.audio_feature_path))
            self.word_feature = self.unroll_and_normalize(self.load_fused_word_feature())
        elif model_type == 'bimodal_model':
            self.visual_feature = self.unroll_and_normalize(self.load_fused_visual_feature())
            self.audio_feature = self.unroll_and_normalize(self.load_feature(self.audio_feature_path))
        elif model_type == 'audio_model':
            self.audio_feature = self.unroll_and_normalize(self.load_feature(self.audio_feature_path))
        elif model_type== 'visual_model':
            self.visual_feature = self.unroll_and_normalize(self.load_fused_visual_feature())
        elif model_type == 'word_model':  
            self.word_feature = self.unroll_and_normalize(self.load_fused_word_feature())
    def load_feature(self, path_list):
        if len(path_list) == 1:
            #only contains train and validation
            with open(path_list[0],'rb') as f:
                return pickle.load(f)
        elif len(path_list) == 2:
            feature = {}
            with open(path_list[0],'rb') as f:
                feature =  pickle.load(f)
            with open(path_list[1],'rb') as f:
                data= pickle.load(f)
                if len(data.keys()) == 1:
                    feature['Test'] = data['Test']
                else:
                    feature['Test'] = data

            return feature

    def load_fused_visual_feature(self):
        visual_f_part0 = self.load_feature(self.visual_feature_path0)
        visual_f_part1 = self.load_feature(self.visual_feature_path1)
        #fuse two parts
        fused_visual_f = {}
        states = visual_f_part0.keys()
        for state in states:
            fused_visual_f[state] = {}
            videos = visual_f_part0[state].keys()
            for vid in videos:
                if vid in visual_f_part1[state].keys():
                    fused_visual_f[state][vid] = {}
                    utters = visual_f_part0[state][vid].keys()
                    for uttr in utters:
                        if uttr in visual_f_part1[state][vid].keys():
                            f0 = visual_f_part0[state][vid][uttr]
                            f1 = visual_f_part1[state][vid][uttr]
                            assert f0.shape[0] == f1.shape[0]
                            fused_visual_f[state][vid][uttr] = np.concatenate((f0,f1), axis = -1)
        return fused_visual_f     
    def load_fused_word_feature(self):
        word_f_part0 = self.load_feature(self.word_feature_path0)
        word_f_part1 = self.load_feature(self.word_feature_path1)
        #fuse two parts
        fused_word_f = {}
        states = word_f_part0.keys()
        for state in states:
            fused_word_f[state] = {}
            videos = word_f_part0[state].keys()
            for vid in videos:
                if vid in word_f_part1[state].keys():
                    fused_word_f[state][vid] = {}
                    utters = word_f_part0[state][vid].keys()
                    for uttr in utters:
                        if uttr in word_f_part1[state][vid].keys():
                            f0 = np.asarray(word_f_part0[state][vid][uttr])
                            f1 = np.asarray(word_f_part1[state][vid][uttr])
        
                            fused_word_f[state][vid][uttr] = np.concatenate((f0,f1), axis = -1)
        return fused_word_f  
    def process_sequence(self, list):
        # make the sequence length is self.seq_length
        length = len(list)
        assert length > 0
        if length >= self.seq_length:
            return list[:self.seq_length]
        else:
            for _ in range(self.seq_length-length):
                list.append(np.zeros(list[0].shape))
            return list
    def get_audio_feature(self,vid,uttr, mode):
        state_name = mode
        try:
            utter_feature = self.audio_feature[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in audio_feature!')
            return None
        else:
            
            return utter_feature

    def get_visual_sequence_from_visual_feature(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.visual_feature[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in visual_feature!')
            utter_feature = None
            
        return utter_feature
    def get_word_feature(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.word_feature[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in word_feature!')
            return None
        else:
            
            return utter_feature
    def get_label(self, train_valid_test, vid, uttr):
        #according to task, return arousal, valence or emotion category
        if self.task == 'arousal':
            return self.data[train_valid_test][vid][uttr][0]
        elif self.task == 'valence':
            return self.data[train_valid_test][vid][uttr][1]
        else:
            return self.data[train_valid_test][vid][uttr][2]

    def get_all_sequences_in_memory(self, train_valid_test):
        """
        :param train_valid_test:
        :param task_type: 'arousal','valence' or 'emotion'
        :param feature_type: 'visual','arousal','word', 'bimodal','trimodal'
        :return:
        """
        
        data = self.data
        
        if train_valid_test in ['Train','Validation','Test']:
            main_dict = data[train_valid_test]
            print("Loading %s dataset with %d videos."%(train_valid_test, len(main_dict.keys())))
            x_audio = []
            x_visual = []
            x_word = []
            name_list = []
            y = []

            videos = main_dict.keys()
            if self.model == 'trimodal_model':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        a = self.get_audio_feature(vid, uttr, train_valid_test)
                        b = self.get_visual_sequence_from_visual_feature(vid, uttr, train_valid_test)
                        c = self.get_word_feature(vid, uttr, train_valid_test)

                        if (a is not None) and (b is not None) and (c is not None) :
                            x_audio.append(a)
                            x_visual.append(b)
                            x_word.append(c)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x = [np.asarray(x_audio), np.asarray(x_visual), np.asarray(x_word)]
                if self.istrain:
                    y = np.asarray(y)
                
            elif self.model == 'audio_model':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        a = self.get_audio_feature(vid, uttr, train_valid_test)

                        if (a is not None) :
                            x_audio.append(a)

                            y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_audio)
                y = np.asarray(y)
            elif self.model == 'bimodal_model':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        a = self.get_audio_feature(vid, uttr, train_valid_test)
                        b = self.get_visual_sequence_from_visual_feature(vid, uttr, train_valid_test)

                        if (a is not None)  :
                            x_audio.append(a)
                            x_visual.append(b)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x =[np.asarray(x_audio), np.asarray(x_visual)]
                if self.istrain:
                    y = np.asarray(y)  
            elif self.model == 'visual_model':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        b = self.get_visual_sequence_from_visual_feature(vid, uttr, train_valid_test)

                        if (b is not None) :
                            x_visual.append(b)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x =  np.asarray(x_visual)
                if self.istrain:
                    y = np.asarray(y)
                
            elif self.model == 'word_model':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        c = self.get_word_feature(vid, uttr, train_valid_test)

                        if (c is not None) :
                            x_word.append(c)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_word)
                if self.istrain:
                    y = np.asarray(y)
            if self.istrain:
                return x, y, name_list
            else:
                return x, name_list









