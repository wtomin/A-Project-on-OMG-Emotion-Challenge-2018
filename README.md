# A-Project-on-OMG-Emotion-Challenge-2018
This is the code repository for the OMG emotion challenge 2018.                    

## Prerequisite
To run this code, you need to install these libraries first
 + Keras 2.+
 + [keras-vggface 0.5 ](https://github.com/rcmalli/keras-vggface)
 + tensorflow 1.5+ 
 + [OpenFace 1.0.0](https://github.com/TadasBaltrusaitis/OpenFace)
 + [openSMILE feature extraction tool](https://github.com/naxingyu/opensmile)
 + nltk    3.+
 + pickle, numpy, sklearn, matplotlib, csv, matlab

## Instructions
In **data prepration**, all videos will be downloaded, andsplitted into utterances into /Videos/Train, /Videos/Validation,/Video/Test
(the csv files for train, validation, test set can be requested from [OMG emotion challenge](https://www2.informatik.uni-hamburg.de/wtm/OMG-EmotionChallenge/))
1. data_preparation: run `python create_videoset.py`

In **feature extraction**, the features for three modal are extracted

2. feature_extraction:<br>
   - run `python OpenFace_extractor`: [OpenFace features](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format) are extracted<br>
   - run `python generate_visual_features.py`: [VGG Face fc6 features](https://github.com/rcmalli/keras-vggface) are extracted.<br>
   - run `extract_audio_files.py`: audio files are extracted from video format files.<br>
   - run `generate_audio_feature_utterance_level.py`: [openSMILE features](https://github.com/naxingyu/opensmile/blob/master/config/emobase2010.conf) are extracted.<br>
   - run `generate_word_features.py`: text features from [Bing Liu's opinion lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) are extracted.<br>
   - run `count_pn.m` : text features from [MPQA Subjectivity Lexicon](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/) are extracted.
   
In **experiment**:<br>
  - `data.py` provides normalized features and labels. <br>
  - `models.py` contains definitions of unimodal models, trimodal models in late and early fusion. <br>
  - `functions.py` defines some custom functions used as loss function or metric.<br>
  - `train.py`: train and evaluation.<br>

## Multimodal Fusion
### Early Fusion
![early fusion](https://github.com/wtomin/A-Project-on-OMG-Emotion-Challenge-2018/blob/master/early_fusion.png)

### Late Fusion
![late fusion](https://github.com/wtomin/A-Project-on-OMG-Emotion-Challenge-2018/blob/master/late_fusion.png)
