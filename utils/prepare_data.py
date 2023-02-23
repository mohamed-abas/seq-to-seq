import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

'''
def prep_csv(input_files, file_path, save_file, save_path, new_file_name):
     the function will generate a single dataframe for all given files 
     input_files: should be a list of tuples 
     tuple(antigenic data as csv file, protien data as fasta sequence file, H value, N Value)
     file_path: path to input files
     save_file: bool
     save_path: path to save new dataframe it will be saved as csv file
     new_file_name: new file name for generated_dataframe
     pad_max_width: integer of the width to pad all virus sequences to
    DF_list = list()
    for i in range(0, len(input_files)):
        antigenic_data_file = file_path + '/' + input_files[i][0]
        sequence_data_file = file_path + '/' + input_files[i][1]
        # H = input_files[i][2]
        # N = input_files[i][3]
        antigenic_data = pd.read_csv(antigenic_data_file)
        # antigenic_data['H'] =  H
        # antigenic_data['N'] =  N
        seq =list()
        
        for record in SeqIO.parse(sequence_data_file, "fasta"):
            seq.append([record.id, str(record.seq)]) 
        sequence_data = pd.DataFrame(data= seq, columns=['virus_id','virus_seq'])
        sequence_data.drop_duplicates(keep='first', inplace= True)
        flu_step1 = pd.merge(left = antigenic_data, right = sequence_data, how='left', left_on='virusA', right_on='virus_id')
        flu_step1 = flu_step1.rename(columns={'virus_id': 'virusA_id', 'virus_seq': 'virusA_seq'})
        flu_step2 = pd.merge(left = flu_step1, right = sequence_data, how='left', left_on='virusB', right_on='virus_id')
        flu_step2 = flu_step2.rename(columns={'virus_id': 'virusB_id', 'virus_seq': 'virusB_seq'})
        flu_step2 = flu_step2[['virusA','virusB','virusA_seq','virusB_seq','Antigenic_Distance']]
        DF_list.append(flu_step2)   
    flu_data = pd.concat(DF_list, ignore_index=True)
    # flu_data.virusA_seq = flu_data.virusA_seq.str.pad(pad_max_width, 'right',fillchar='*')
    # flu_data.virusB_seq = flu_data.virusA_seq.str.pad(pad_max_width, 'right',fillchar='*')
    flu_data.to_csv(save_path + '/' + new_file_name ,index = False)
    return flu_data
'''

class preproccess_data:
    def __init__(self,
                 flu_csv_file = 'prepared_data/flu_data_x2.csv',
                 no_of_char_per_word = 1,
                 seq_cols_list = ['virusA_seq', 'virusB_seq'],
                 input_col = 'virusB_seq',
                 target_col = 'virusA_seq',
                 antigenic_distance_col = 'Antigenic_Distance',
                 embedding_size = 50,
                 add_begin_end = True,
                 randomize_df = False,
                 antigenic_distance_scale = 'standard',
                 test_set_size = .2
                 ):
                 
        self.flu_csv_file = flu_csv_file
        self.no_of_char_per_word = no_of_char_per_word
        self.seq_cols_list = seq_cols_list
        self.input_col = input_col
        self.target_col = target_col
        self.antigenic_distance_col = antigenic_distance_col
        self.add_begin_end = add_begin_end
        self.randomize_df = randomize_df
        self.embedding_size = embedding_size
        self.antigenic_distance_scale = antigenic_distance_scale
        self.test_set_size = test_set_size
        
        self.input_virus_seq_set = None
        self.input_max_seq_len = None
        self.input_words = None
        self.input_num_tokens = None
        self.input_token_index = None
        self.input_reverse_index = None
        
        self.target_virus_seq_set = None
        self.target_max_seq_len = None
        self.target_words = None
        self.target_num_tokens = None
        self.target_token_index = None
        self.target_reverse_index = None   
        
        self.input_virus_seq_set = None 
        self.input_words = None
        self.target_virus_seq_set = None
        self.target_words = None
        
        self.zero_encoder_input_data = None
        self.zero_decoder_input_data = None
        self.zero_decoder_target_data = None
        self.zero_enc_antigenic_distance = None
        self.zero_dec_antigenic_distance = None
        
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None
        self.no_records_target = None
        self.no_records_input = None
        
        self.enc_antigenic_distance = None
        self.dec_antigenic_distance = None
        
        self.enc_antigenic_distance_test = None
        self.enc_antigenic_distance_train = None 
        self.encoder_input_data_test = None
        self.encoder_input_data_train = None
        self.decoder_input_data_test = None
        self.decoder_input_data_train = None
        self.decoder_target_data_test = None
        self.decoder_target_data_train = None      
        
        self.flu_df = None
        self.all_index = None
        self.train_index = None
        self.test_index = None
        
        
    def get_flu_df(self):
        self.flu_df = pd.read_csv(self.flu_csv_file)
        self.no_records_target = len(self.flu_df)
        self.no_records_input = len(self.flu_df)
        
    def seq_to_words(self):
        '''
        this function will take a data frame , target column name and no of chars in a word
        then divides a sequence to a list of words
        add_begin_end for adding begin and end symbols of a sequence
        randomize_df to randomize dataframe rows
        flu_df: the resulting data frame from prep_csv function
        col_name: list of columns names which are holding a virus sequence
        no_of_char: number of charachters per word 
        '''
        #lmda_sep = lambda a: [a[i:i+no_of_char] for i in range(0, len(a), no_of_char)]
        lmda_sep = lambda a: ' '.join([a[i:i+self.no_of_char_per_word] for i in range(0, len(a), self.no_of_char_per_word)])
        for i in range(0,len(self.seq_cols_list)):
            self.flu_df[self.seq_cols_list[i]] = self.flu_df[self.seq_cols_list[i]].map(lmda_sep) 
        if self.add_begin_end:
           self.flu_df[self.target_col] = self.flu_df[self.target_col].apply(lambda x : '<BOS> '+ x + ' <EOS>')
        if self.randomize_df:
           self.flu_df = self.flu_df.sample(frac = 1)
           self.flu_df = self.flu_df.reset_index(drop = True)     

    def create_tokens(self, column_name):
      '''
       this function will take a data frame and a column name 
       the function will return 
       virus sequence set (vocabulary)
       maximum sequence length(max number of words in a sequence)
       words is virus sequence vocabulary as a list
       number of tokens in vocabulary list
       token index is a dictinary indexing of each word of the vocabulary
       reverse index  is a dictinary where each word is a key and the word index is the value
      '''
      virus_seq_set=set()
      for virus_seq in self.flu_df[column_name]:
          for word in virus_seq.split():
              if word not in virus_seq_set:
                  virus_seq_set.add(word)
      len_virus_seq_list=[]
      for l in self.flu_df[column_name]:
          len_virus_seq_list.append(len(l.split(' ')))
      max_seq_len = np.max(len_virus_seq_list)
      words = sorted(list(virus_seq_set))
      num_tokens = len(virus_seq_set)
      token_index = dict([(word, i) for i, word in enumerate(words)])
      reverse_index = dict((i, char) for char, i in token_index.items())
      return virus_seq_set, max_seq_len, words, num_tokens, token_index, reverse_index 

    def create_zero_matrix(self):
      '''
      this function takes 
      number of records in target sequence
      number of records in input sequence
      maximum legnth for input sequence 
      maximum legnth for target sequence
      number of decoder tokens
      embedding size
      then the function will create zero matrix for 
      encoder_input_data
      decoder_input_data
      enc_antigenic_distance
      dec_antigenic_distance
      '''
      self.zero_encoder_input_data = np.zeros((self.no_records_input, self.input_max_seq_len), dtype='float32')
      self.zero_decoder_input_data = np.zeros((self.no_records_target, self.target_max_seq_len),dtype='float32')
      self.zero_decoder_target_data = np.zeros((self.no_records_target, self.target_max_seq_len, self.target_num_tokens), dtype='float32')
      self.zero_enc_antigenic_distance = np.zeros((self.no_records_input, self.input_max_seq_len, self.embedding_size), dtype='float32')
      self.zero_dec_antigenic_distance = np.zeros((self.no_records_target, self.target_max_seq_len, self.embedding_size), dtype='float32')
      
    def create_feature_matrix(self):
      '''
      the function will take a dataframe, input column name, target column name  
      function will fill the zero matrix for
      encoder_input_data
      decoder_input_data
      decoder_target_data
      and return the each matrix with corresponding values
      '''
      self.encoder_input_data = self.zero_encoder_input_data
      self.decoder_input_data = self.zero_decoder_input_data
      self.decoder_target_data = self.zero_decoder_target_data
      for i, (input_text, target_text) in enumerate(zip(self.flu_df.virusB_seq, self.flu_df.virusA_seq)):
        for t, word in enumerate(input_text.split()):
            self.encoder_input_data[i, t] = self.input_token_index[word]
        for t, word in enumerate(target_text.split()):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            self.decoder_input_data[i, t] = self.target_token_index[word]
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                self.decoder_target_data[i, t - 1, self.target_token_index[word]] = 1. 

    def create_scale_antigenic_distance(self):
      '''
      this function takes zero matrix for 
      enc_antigenic_distance
      dec_antigenic_distance
      maximum legnth for input sequence 
      maximum legnth for target sequence
      embedding size
      dataframe, input column name for antigenic distance 
      the function will return input and target antigenic distances scaled using
      either standard or MinMax scaler default is standard 
      '''

      self.enc_antigenic_distance = self.zero_enc_antigenic_distance
      self.dec_antigenic_distance = self.zero_dec_antigenic_distance
      ad = self.flu_df[self.antigenic_distance_col]  # Antigenic_Distance
      if self.antigenic_distance_scale == 'standard':
        scaler = StandardScaler()
      elif self.antigenic_distance_scale == 'minmax':
        scaler = MinMaxScaler()   
      ad_scaler = scaler.fit(self.flu_df[self.antigenic_distance_col].values.reshape(-1,1))
      ad_scaled = scaler.transform(self.flu_df[self.antigenic_distance_col].values.reshape(-1,1))

      ad_scaled = self.flu_df[self.antigenic_distance_col].values.reshape(-1,1)
      word_ad = np.zeros(self.embedding_size , dtype='float32')
      statement_ad = np.zeros((self.input_max_seq_len, self.embedding_size) , dtype='float32')
      for i in range(0, ad_scaled.shape[0]):
          word_ad[0:self.embedding_size] = ad_scaled[i][0]
          statement_ad[0:self.input_max_seq_len] = word_ad
          self.enc_antigenic_distance[i] =  statement_ad

      word_ad = np.zeros(self.embedding_size , dtype='float32')
      statement_ad = np.zeros((self.target_max_seq_len, self.embedding_size) , dtype='float32')   
      for i in range(0, ad_scaled.shape[0]):
          word_ad[0:self.embedding_size] = ad_scaled[i][0]
          statement_ad[0:self.target_max_seq_len] = word_ad
          self.dec_antigenic_distance[i] =  statement_ad

    def split_train_test(self):
        '''
        wraper function for sklearn.train_test_split
        the function will split data based on indexes
        '''
        self.all_index = list(range(0,len(self.flu_df)))
        self.train_index, self.test_index = train_test_split(self.all_index, test_size = self.test_set_size, stratify = self.flu_df.virusA_seq.str[0:11])        
        self.enc_antigenic_distance_test = self.enc_antigenic_distance[self.test_index]
        self.enc_antigenic_distance_train = self.enc_antigenic_distance[self.train_index] 
        self.encoder_input_data_test = self.encoder_input_data[self.test_index]
        self.encoder_input_data_train = self.encoder_input_data[self.train_index]
        self.decoder_input_data_test = self.decoder_input_data[self.test_index]
        self.decoder_input_data_train = self.decoder_input_data[self.train_index]
        self.decoder_target_data_test = self.decoder_target_data[self.test_index]
        self.decoder_target_data_train = self.decoder_target_data[self.train_index]

    def preproc_data(self):          
        self.get_flu_df()
        
        self.seq_to_words()
        
        self.input_virus_seq_set, self.input_max_seq_len, self.input_words,\
        self.input_num_tokens, self.input_token_index, self.input_reverse_index = self.create_tokens(self.input_col)
        
        self.target_virus_seq_set, self.target_max_seq_len, self.target_words,\
        self.target_num_tokens, self.target_token_index, self.target_reverse_index = self.create_tokens(self.target_col)
        
        self.create_zero_matrix()
        
        self.create_feature_matrix()
        
        self.create_scale_antigenic_distance()
        
        self.split_train_test()