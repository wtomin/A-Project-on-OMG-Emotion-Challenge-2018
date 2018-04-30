from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import load_model
from models import ResearchModels
from data import DataSet
import tensorflow as tf
import time
import os.path
import pdb
import functions
import numpy as np
import sys
sys.path.append('..')
from calculateEvaluationCCC import ccc, mse, f1
from utils import  display_true_vs_pred, print_out_csv
from subprocess import call
import subprocess
# define hyperparameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model", "trimodal_model", "the chosen model,should be one of : visual_model, audio_model, word_model, trimodal_model.")
tf.app.flags.DEFINE_string("pretrained_model_path",None, "the pretrained_model_path.")
tf.app.flags.DEFINE_string("task","valence", "The regression task for arousal, valence or emotion categories.")
tf.app.flags.DEFINE_boolean("is_train", True, "True for training, False for evaluation.")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "The initial learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 300, "The number of utterances in each batch.")
tf.app.flags.DEFINE_integer("nb_epochs", 300, "Number of epochs for training.")
#for reproductivity
np.random.seed(123)
def load_custom_model(pretrained_model_path):
    model = load_model(pretrained_model_path, 
                           custom_objects ={'ccc_metric':functions.ccc_metric,
                                            'ccc_loss': functions.ccc_loss})
    return model

def train(istrain = True,model='visual_model', saved_model_path=None, task = 'arousal',
         batch_size = 2, nb_epoch=200, learning_r = 1e-3):
    """
    train the model
    :param model: 'visual_model','audio_model','word_model','trimodal_model'
    :param saved_model_path: saved_model path
    :param task: 'aoursal','valence','emotion'
    :param batch_size: 2
    :param nb_epoch:2100
    :return:s
    """
    timestamp =  time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
    # Helper: Save the model.
    if not os.path.exists(os.path.join('checkpoints', model)):
        os.makedirs(os.path.join('checkpoints', model))
    checkpointer = ModelCheckpoint(
        #filepath = os.path.join('checkpoints', model, task+'-'+ str(timestamp)+'-'+'best.hdf5' ),
        filepath = os.path.join('checkpoints', model, task+'-'+ str(timestamp)+'-'+'best.hdf5' ),
        verbose=1,
        save_best_only=True)
    
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20)
    
    # Helper: Save results.
    
    csv_logger = CSVLogger(os.path.join('logs', model , task +'-'+ \
        str(timestamp) + '.log'))

    # Get the data and process it.
    # seq_length for the sentence
    seq_length = 20
    dataset = DataSet(
        istrain = istrain,
        model = model,
        task = task,
        seq_length=seq_length
        )

    # Get the model.
    rm = ResearchModels(
            istrain = istrain,
            model = model, 
            seq_length = seq_length, 
            saved_model_path=saved_model_path, 
            task_type= task,
            saved_audio_model = None,
            saved_visual_model = None,
            saved_word_model = None,
            learning_r = learning_r
            )
    # Get training and validation data.
    x_train, y_train, train_name_list = dataset.get_all_sequences_in_memory('Train')
    x_valid, y_valid, valid_name_list= dataset.get_all_sequences_in_memory('Validation')
    
    # Fit!
    # Use standard fit.
    rm.model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_valid,y_valid),
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger,  checkpointer],
        #callbacks=[tb, lrate, csv_logger,  checkpointer],
        epochs=nb_epoch)
    
    # find the current best model and get its prediction on validation set
    model_weights_path = os.path.join('checkpoints', model, task+'-'+ str(timestamp)+'-'+'best.hdf5' )
    
    best_model= load_custom_model(model_weights_path)
    

    y_valid_pred = best_model.predict(x_valid)
    y_valid_pred = np.squeeze(y_valid_pred)
    
    y_train_pred = best_model.predict(x_train)
    y_train_pred = np.squeeze(y_train_pred)

    #calculate the ccc and mse
    if task in ['arousal', 'valence']:
        print("The CCC in validation set is {}".format(ccc(y_valid, y_valid_pred)[0]))
        print("The mse in validation set is {}".format(mse(y_valid, y_valid_pred)))
        
        print("The CCC in train set is {}".format(ccc(y_train, y_train_pred)[0]))
        print("The mse in train set is {}".format(mse(y_train, y_train_pred)))
    elif task == "emotion":
        print("F1 score in validation set is {}".format(f1(y_valid, y_valid_pred)))
    # display the prediction and true label
    log_path = os.path.join('logs', model , task +'-'+ \
        str(timestamp) + '.log')
    
    display_true_vs_pred([y_valid,y_train], [y_valid_pred, y_train_pred],log_path, task, model)

def evaluate_on_test(arousal_model_path, valence_model_path, output_file,istrain = False ):
    arousal_model = load_custom_model(arousal_model_path)
    valence_model = load_custom_model(valence_model_path)
    model = 'trimodal_model'
    dataset = DataSet(
        istrain = istrain,
        model = model
        )
    
    #load test data
    x_test, name_list = dataset.get_all_sequences_in_memory('Test')

    arousal_pred = arousal_model.predict(x_test)
    arousal_pred = np.squeeze(arousal_pred)
    valence_pred = valence_model.predict(x_test)
    valence_pred = np.squeeze(valence_pred)
    
    print_out_csv(arousal_pred, valence_pred, name_list, '../omg_TestVideos.csv', output_file)
    
def evaluate_on_validation(arousal_model_path, valence_model_path, output_file,istrain = True):
    arousal_model = load_custom_model(arousal_model_path)
    valence_model = load_custom_model(valence_model_path)
    model = 'trimodal_model'
    dataset = DataSet(
    istrain = istrain,
    model = model,
    )
    x_valid, y_valid, valid_name_list= dataset.get_all_sequences_in_memory('Validation')
    
    arousal_pred = arousal_model.predict(x_valid)
    arousal_pred = np.squeeze(arousal_pred)
    valence_pred = valence_model.predict(x_valid)
    valence_pred = np.squeeze(valence_pred)
    
    print_out_csv(arousal_pred, valence_pred, valid_name_list, '../omg_ValidationVideos.csv', output_file)
    
    cmd = 'python ../calculateEvaluationCCC.py ../omg_ValidationVideos_pred.csv ../new_omg_ValidationVideos.csv'
    process = subprocess.Popen(cmd.split(),stderr= subprocess.STDOUT,universal_newlines=True)
    process.communicate()
def main():
    pdb.set_trace()
    if FLAGS.is_train:
        train(istrain=FLAGS.is_train, model = FLAGS.model, saved_model_path=FLAGS.pretrained_model_path, task = FLAGS.task,
            batch_size=FLAGS.batch_size, nb_epoch=FLAGS.nb_epochs, learning_r = FLAGS.learning_rate)
        """
        arousal_model_path = ''
        valence_model_path = ''
        output_file = '../omg_ValidationVideos_pred.csv'
        evaluate_on_validation(arousal_model_path, valence_model_path, output_file,istrain=True)
        """
    else:
        arousal_model_path = ''
        valence_model_path = ''
        output_file = ''
        evaluate_on_test(arousal_model_path, valence_model_path, output_file,istrain=False)

if __name__ == '__main__':
    main()

