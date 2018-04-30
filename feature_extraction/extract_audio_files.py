import csv
import glob
import os
import os.path
from subprocess import call
import subprocess

import pdb
import numpy as np
from tqdm import tqdm

def read_csv(filename):
    data = []
    with open(filename,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append( [row['video'], row['utterance']])
    return data
def save_csv(obj, filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(obj)
        
def check_already_extracted(istrain,video, uttr_name):
    des = os.path.join('../Audio_Frames',istrain, video,uttr_name+'.wav')
    return os.path.exists(des)
             
def extract_audio_frames(csv_file, istrain):
    ids = read_csv(csv_file)
    error_report = []
    data_file = [['Istrain', 'VideoName', 'Utterance_index','Num_frames','Frame_index','Frame_path']]
    # Train or Validation
    folder = istrain
    print("Extracting for "+istrain+" dataset...")
    for id in tqdm(ids):
        pass
        video,uttr = id
        main_folder = os.path.join('../Videos',folder)
        video_path = os.path.join(main_folder, video)
        if os.path.exists(video_path):
            
            uttr_path = os.path.join(video_path, uttr)
            if os.path.exists(uttr_path):
                uttr_name = uttr.split('.')[0]
                if not check_already_extracted(istrain,video,uttr_name):
                    print(video+': '+uttr+" is extracting...")
                    src = uttr_path
                    des = os.path.join('../Audio_Frames', istrain, video,uttr_name+'.wav') # the total mp3 file path
                    
                    if not os.path.exists(os.path.join('../Audio_Frames')):
                        os.mkdir(os.path.join('../Audio_Frames'))
                    if not os.path.exists(os.path.join('Audio_Frames', istrain)):
                        os.mkdir(os.path.join('../Audio_Frames', istrain))
                    if not os.path.exists(os.path.join('../Audio_Frames', istrain,video)):
                        os.mkdir(os.path.join('../Audio_Frames', istrain,video))
                    if not os.path.exists(os.path.join('../Audio_Frames', istrain,video, uttr_name)):
                        os.mkdir(os.path.join('../Audio_Frames', istrain,video, uttr_name))
                    process1 = subprocess.call(['ffmpeg', '-i', src, '-vn', des]) # extract audio file from video file"""
                    #no need for test set
                    # then segment the mp3 file into same length interval
                    audio_interval = 0.5 # seconds
                    des2 = os.path.join('../Audio_Frames', istrain,video, uttr_name, 'out%04d.wav')
                    process2 = subprocess.call(['ffmpeg','-i',des, '-f','segment','-segment_time', str(audio_interval),des2])
                extracted_uttr = os.path.join('../Audio_Frames', istrain,video, uttr_name)
                frames = sorted(glob.glob(os.path.join(extracted_uttr,'*.wav')))
                Num_frames = len(frames)
                for i, path in enumerate(frames):
                    uttr_index = uttr_name.split('_')[-1]
                    data_file.append([istrain,video, uttr_index, Num_frames, str(i+1),path])   
            else:
                print(video+': '+uttr+"doesnt exist")
                error_report.append(video+': '+uttr+"doesnt exist\n")
        else:
            print(video+"doesnt exist")
            error_report.append(video+"doesnt exist\n")
            
   
def extract_audio_frames_test(csv_file, istrain):
    ids = read_csv(csv_file)
    # Test
    folder = istrain
    error_report = []
    print("Extracting for "+istrain+" dataset...")
    for id in tqdm(ids):
        pass
        video,uttr = id
        main_folder = os.path.join('../Videos',folder)
        video_path = os.path.join(main_folder, video)
        if os.path.exists(video_path):
            #print(video+"is extracting now...")
            uttr_path = os.path.join(video_path, uttr)
            if os.path.exists(uttr_path):
                uttr_name = uttr.split('.')[0]
                if not check_already_extracted(istrain,video,uttr_name):
                    
                    src = uttr_path
                    des = os.path.join('../Audio_Frames', istrain, video,uttr_name+'.wav') # the total mp3 file path
                    #check whether des exists
                    if not os.path.exists(os.path.join('../Audio_Frames')):
                        os.mkdir(os.path.join('../Audio_Frames'))
                    if not os.path.exists(os.path.join('Audio_Frames', istrain)):
                        os.mkdir(os.path.join('../Audio_Frames', istrain))
                    if not os.path.exists(os.path.join('Audio_Frames', istrain,video)):
                        os.mkdir(os.path.join('../Audio_Frames', istrain,video))
                    if not os.path.exists(os.path.join('../Audio_Frames', istrain,video, uttr_name)):
                        os.mkdir(os.path.join('../Audio_Frames', istrain,video, uttr_name))
                    process1 = subprocess.call(['ffmpeg', '-i', src, '-vn', des]) # extract audio file from video file
                     
            else:
                print(video+': '+uttr+"doesnt exist")
                error_report.append(video+': '+uttr+"doesnt exist\n")

        else:
            print(video+"doesnt exist")
            error_report.append(video+"doesnt exist\n")
    print(error_report)
    
def extract_all_frames():
    extract_audio_frames('../omg_TrainVideos.csv', 'Train')
    extract_audio_frames('../omg_ValidationVideos.csv', 'Validation')   
    
def main():

    #extract_all_frames()
    extract_audio_frames_test('../omg_TestVideos_WithoutLabels.csv','Test')

if __name__ == '__main__':
    pdb.set_trace()
    main()

