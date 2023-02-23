from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.pairwise2 import format_alignment
from nltk.translate.bleu_score import sentence_bleu


class evaluate_s2s:
    def __init__(self,               
                 test_index,
                 max_len_decoded_sentence,
                 no_recorde_to_evaluate,
                 encoder_input_data_test,
                 enc_antigenic_distance_test,
                 target_token_index,
                 reverse_target_index,
                 seq_model,
                 results_df_file_name,
                 summary_df_file_name,
                 real_vireus_df_col,
                 real_virus_name_df_col
                 ):
                 
        
        self.test_index = test_index
        
         
        self.no_recorde_to_evaluate = no_recorde_to_evaluate
        self.encoder_input_data_test = encoder_input_data_test
        self.enc_antigenic_distance_test = enc_antigenic_distance_test
        self.real_vireus_df_col = real_vireus_df_col
        self.target_token_index = target_token_index
        self.reverse_target_index = reverse_target_index
        self.seq_model = seq_model
        self.results_df_file_name = results_df_file_name
        self.summary_df_file_name = summary_df_file_name
        self.real_virus_name_df_col = real_virus_name_df_col
        self.real_virus_name = None
        self.generated_seq = None
        self.real_virus = None
        
        self.alignment_score = None
        self.score_one_gram = None
        self.score_two_gram = None
        self.score_three_gram = None
        self.score_four_gram = None
        self.cum_bleu_score = None
        self.decoded_sentence = None
        self.max_len_decoded_sentence = max_len_decoded_sentence
    
    def pred_virus_align_score(self):
        virus_real = self.real_virus
        virus_real = virus_real.replace("<BOS>", "")
        virus_real = virus_real.replace("<EOS>", "")
        virus_real = virus_real.replace(" ", "")
        virus_generated = self.generated_seq.replace("<EOS>", "")
        virus_generated = virus_generated.replace(" ", "")
        alignments = pairwise2.align.globalxx(virus_real, virus_generated)
        self.alignment_score = alignments[-1][2]/len(virus_real)
        
    def pred_virus_bleu_score(self):
        virus_real = self.real_virus
        virus_real = virus_real.replace("<BOS>", "")
        virus_real = virus_real.replace("<EOS>", "") 
        virus_generated = self.generated_seq.replace("<EOS>", "")
        self.score_one_gram = sentence_bleu([virus_real.split()], virus_generated.split(), weights=(1, 0, 0, 0))
        self.score_two_gram = sentence_bleu([virus_real.split()], virus_generated.split(), weights=(0, 1, 0, 0))
        self.score_three_gram = sentence_bleu([virus_real.split()], virus_generated.split(), weights=(0, 0, 1, 0))
        self.score_four_gram = sentence_bleu([virus_real.split()], virus_generated.split(), weights=(0, 0, 0,1))
        self.cum_bleu_score = sentence_bleu([virus_real.split()], virus_generated.split(), weights=(0.25, 0.25, 0.25, 0.25))
                

    def evaluate_model(self):
        # max_len_decoded_sentence = (3 * max_virusA_seq_len) + max_virusA_seq_len - 1  # to be taken from outside the class
        
        res_lst = []
        for seq_index in range(0,self.no_recorde_to_evaluate): # len(test_index)
            input_seq =  self.encoder_input_data_test[seq_index:seq_index+1]
            input_ad =   self.enc_antigenic_distance_test[seq_index:seq_index+1]
            self.real_virus = self.real_vireus_df_col[self.test_index[seq_index]]
            self.real_virus_name = self.real_virus_name_df_col[self.test_index[seq_index]]
            decoded_sentence = self.seq_model.decode_sequence(input_seq, input_ad, self.max_len_decoded_sentence, self.target_token_index, self.reverse_target_index)      
            self.generated_seq = decoded_sentence
            self.pred_virus_align_score()
            self.pred_virus_bleu_score()
            #print(self.real_virus_name)
            res_lst.append([
                             self.test_index[seq_index],
                             self.real_virus_name,
                             self.real_virus.replace("<EOS>", "").replace("<BOS>", "").replace(" ",""), 
                             self.generated_seq.replace("<EOS>", "").replace(" ",""), 
                             self.alignment_score, 
                             self.score_one_gram, 
                             self.score_two_gram, 
                             self.score_three_gram, 
                             self.score_four_gram, 
                             self.cum_bleu_score   
                            ])

        # resuls save wriiten by fathallah
        self.seq_model.set_results_df(res_lst, dump_df = self.results_df_file_name) 

        # for each metric in the "get_statistical_results" you can plot an error_bar/IQR of the mean and std
        self.seq_model.get_statistical_results(dump_df = self.summary_df_file_name)    
