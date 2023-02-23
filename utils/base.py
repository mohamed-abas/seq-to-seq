from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Add, Concatenate, Dense, Dropout, LSTM, GRU, Embedding, Bidirectional, TimeDistributed, AdditiveAttention
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import matplotlib.pyplot as plt
import pickle, traceback
import os
import numpy as np
import pandas as pd

from .attention import AttentionLayer
from .lr_decay import step_decay_schedule
from ..time_utils import get_TimeStamp_str
from ..pkl_utils import dump, load_pkl

class base_model:
    def __init__(self, exp_title,
                 loss='categorical_crossentropy',
                 metrics=['acc'],
                 recurrent_layer='LSTM',
                 embedding_size=50,
                 num_encoder_tokens=None,
                 num_hidden_cells=None,
                 num_decoder_tokens=None,
                 dropout=0.2,
                 optimizer='adam',
                 lr=.001,
                 decay_step_size=20,
                 decay_factor=.98,
                 max_target_seq_len=327,

                 ):
        self.loss = loss
        self.metrics = metrics
        self.recurrent_layer = recurrent_layer
        self.embedding_size = embedding_size
        self.num_encoder_tokens = num_encoder_tokens
        self.num_hidden_cells = num_hidden_cells
        self.num_decoder_tokens = num_decoder_tokens
        self.dropout = dropout
        self.optimizer = optimizer
        self.lr = lr
        self.decay_step_size = decay_step_size
        self.decay_factor = decay_factor
        self.max_target_seq_len = max_target_seq_len
        self.exp_title = exp_title
        
        self.model=None,
        self.encoder_inputs=None,
        self.enc_ad_input=None,
        self.encoder_states=None,
        self.decoder_inputs=None,
        self.dex=None,
        self.decoder_lstm=None,
        self.decoder_dense_softmax=None,
        self.attention=None,
        self.encoder_outputs=None,
        self.decoder_outputs=None,
        self.res_df = None
        self.summary = None
        self.encoder_model=None
        self.decoder_model=None
        self.history = None

        self.result_df_columns = ['test_index',
                                  'real_virus_name',
                                  'virus_real',
                                  'virus_generated',
                                  'alignment_score',
                                  'score_one_gram',
                                  'score_two_gram',
                                  'score_three_gram',
                                  'score_four_gram',
                                  'cum_bleu_score']

        self.exp_id = (get_TimeStamp_str() + ' ' + exp_title).strip()
        os.mkdir(self.exp_id)

    def build(self):
        pass

    def inf_encoder_model(self):
        pass

    def inf_decoder_model(self):
        pass

    def decode_sequence(self,
                        input_seq, enc_input_ad, max_len_decoded_sentence,
                        target_token_index, reverse_target_index
                        ):
        pass

    def train(self, encoder_input_data, decoder_input_data, enc_antigenic_distance, decoder_target_data,
              epochs, batch_size, val_split, patience
              ):
        pass

    def set_results_df(self, results_lst, dump_df=None):
        self.res_df = pd.DataFrame(np.array(results_lst), columns=self.result_df_columns)
        if dump_df is not None:
            self.res_df.to_csv(self.exp_id + '/' + dump_df + '.csv')

    def get_statistical_results(self, dump_df=None):
        if self.res_df is not None:
            self.summary = self.res_df[['alignment_score',
                                        'score_one_gram',
                                        'score_two_gram',
                                        'score_three_gram',
                                        'score_four_gram',
                                        'cum_bleu_score']].describe()

            if dump_df is not None:
                self.summary.to_csv(self.exp_id + '/' + dump_df + '.csv')
        else:
            print("No results dataframe is created yet")

    def plot_history(self, save_plots=True):
        plt.plot(self.history.history['acc'], color = 'green')
        plt.plot(self.history.history['val_acc'], color = 'red')
        plt.title(self.recurrent_layer + ' Model Accuracy')
        plt.yticks([.1, .2, .3, .4, .5, .6, .7, .8, .9])
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='best')
        if save_plots:
            plt.savefig(self.exp_id + '/' + self.recurrent_layer + '_model_accuracy.pdf')
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        
        # summarize history for loss
        plt.plot(self.history.history['loss'], color = 'blue')
        plt.plot(self.history.history['val_loss'], color = 'red')
        plt.title(self.recurrent_layer + ' Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='best')
        if save_plots:
            plt.savefig(self.exp_id + '/' +  self.recurrent_layer + '_model_loss.pdf')
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        
    def dump(self, path):
        path = self.exp_id + '/' + path
        try:
            path = path + '.pkl'
            with open(path, 'wb') as f:
                dump(self, f, pickle.HIGHEST_PROTOCOL)

            print('successfully dump to "%s"' % path)
        except FileNotFoundError:
            print('\n**** Err_Z001 Location not found:\n', traceback.format_exc())
        except Exception as exc:
            print('\n**** Err_Z002 Dumping Experiment Err:\n', traceback.format_exc())

