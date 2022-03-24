from os.path import join
import torch
import pandas as pd
from torch.utils.data import Dataset
from nlpClassifiers import settings
from transformers import BertTokenizer
import itertools
import operator
import nltk
from nltk.corpus import stopwords

class Vocabulary:
    def __init__(self, dataset, stopwords_lang=None):
        self.dataset = dataset
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0
        self.num_sentences = 0
        self.longest_sentence = 0
        self.stopwords_lang = stopwords_lang
        self.stopwords = None

        if self.stopwords_lang:
            nltk.download('stopwords')
            self.stopwords = stopwords.words(self.stopwords_lang)

    def build_vocab(self, data, max_words = 0):
        self.max_words = max_words
        for sent in data:
            self.add_sentence(sent)

        sorted_d = sorted(self.word2count.items(), key=operator.itemgetter(1), reverse=True)
        sorted_dict = {}

        self.num_words = 0
        for tuple in sorted_d:
            word = tuple[0]
            count = tuple[1]
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            sorted_dict[word] = count
            self.num_words += 1

        self.word2count = sorted_dict
        if max_words != 0:
            self.word2index = dict(itertools.islice(self.word2index.items(),max_words))
            self.word2count = dict(itertools.islice(self.word2count.items(),max_words))
            self.index2word = dict(itertools.islice(self.index2word.items(),max_words))


    def add_word(self, word):
        if word not in self.word2count:
            # First entry of word into vocabulary
            if (self.stopwords and word not in self.stopwords) or not self.stopwords:
                self.word2count[word] = 1
            #self.index2word[self.num_words] = word
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)

        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

class BOWTokenizer():
    def __init__(self, data):
        word_to_ix = {}
        for sent in data:
            for word in sent.split():
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        self.word_to_ix = word_to_ix
    
    def add_tokens(self, data):
        for sent in data:
            for word in sent.split():
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

class NLPDataset(Dataset):

    def __init__(self, dataset, subset, maxlen, bert_path=None, labels_dict = None, vocab = None):
        #Store the contents of the file in a pandas dataframe
        self.dataset = dataset
        BASE_PATH_TO_DATASET = {"virtual-operator": settings.PATH_TO_VIRTUAL_OPERATOR_DATA, "agent-benchmark": settings.PATH_TO_AGENT_BENCHMARK_DATA, "mercado-livre-pt": settings.PATH_TO_ML_PT_DATA}
        BASE_PATH_TO_DATASET = {"train": join(BASE_PATH_TO_DATASET[self.dataset], "train.csv"), "val": join(BASE_PATH_TO_DATASET[self.dataset], "val.csv"), "test": join(BASE_PATH_TO_DATASET[self.dataset], "test.csv")}
        FULL_PATH_TO_DATASET = BASE_PATH_TO_DATASET[subset]
        self.df = self.read_data(FULL_PATH_TO_DATASET)#.head(500)

        #Initialize the BERT tokenizer
        if bert_path:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        else:
            if(vocab):
                self.vocab = vocab

        if labels_dict == None:
            self.labels_dict = self.get_label_to_ix()
        else:
            self.labels_dict = labels_dict
        
        self.maxlen = maxlen
        self.num_labels = len(self.labels_dict)

    def read_data(self, filename):
        if self.dataset == "mercado-livre-pt":
            sep=","
        else:
            sep=";"
        data = pd.read_csv(filename, sep=sep, names =['utterance','label'], header=None, dtype={'utterance':str, 'label': str} )
        return data

    def __len__(self):
        return len(self.df)

    def get_label_to_ix(self):
        label_to_ix = {}
        for label in self.df.label:
            if label not in label_to_ix:
                label_to_ix[label]=len(label_to_ix)
        return label_to_ix        

    def get_bow_data(self, sentence, label):
        # create a vector of zeros of vocab size = len(word_to_idx)
        vec = torch.zeros(len(self.vocab.word2index))
        for word in sentence.split():
            if word in self.vocab.word2index:
                vec[self.vocab.word2index[word]] = 1
        return vec, self.labels_dict[label]


    def get_bert_data(self, sentence, label):
        token_ids = self.tokenizer.encode(
                    sentence,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                    # This function also supports truncation and conversion
                    # to pytorch tensors, but we need to do padding, so we
                    # can't use these features :( .
                    #max_length = 128,          # Truncate all sentences.
                    #return_tensors = 'pt',     # Return pytorch tensors.
                    )
        if len(token_ids) < self.maxlen:
            token_ids = token_ids + [self.tokenizer.pad_token_id for _ in range(self.maxlen - len(token_ids))] #Padding sentences
        else:
            token_ids = token_ids[:self.maxlen-1] + [self.tokenizer.sep_token_id] #Prunning the list to be of specified max length

        segments_ids = [1] * len(token_ids)
        #Converting the list to a pytorch tensor
        tokens_ids_tensor = torch.tensor(token_ids) 
        segments_ids_tensor = torch.tensor(segments_ids) 

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()
        return tokens_ids_tensor, segments_ids_tensor, attn_mask, self.labels_dict[label]

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'utterance']
        label = self.df.loc[index, 'label']

        if hasattr(self, 'vocab'):
            return self.get_bow_data(sentence, label)
        else:
            return self.get_bert_data(sentence, label)
            