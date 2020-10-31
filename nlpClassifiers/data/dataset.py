from os.path import join
import torch
import pandas as pd
from torch.utils.data import Dataset
from nlpClassifiers import settings
from transformers import BertTokenizer

class NLPDataset(Dataset):

    def __init__(self, dataset, subset, maxlen, bert_path, labels_dict = None):
        BASE_PATH_TO_DATASET = {"virtual-operator": settings.PATH_TO_VIRTUAL_OPERATOR_DATA, "agent-benchmark": settings.PATH_TO_AGENT_BENCHMARK_DATA, "mercado-livre-pt": settings.PATH_TO_ML_PT_DATA}
        BASE_PATH_TO_DATASET = {"train": join(BASE_PATH_TO_DATASET[dataset], "train.csv"), "val": join(BASE_PATH_TO_DATASET[dataset], "val.csv"), "test": join(BASE_PATH_TO_DATASET[dataset], "test.csv")}
        FULL_PATH_TO_DATASET = BASE_PATH_TO_DATASET[subset]
        #Store the contents of the file in a pandas dataframe
        self.df = self.read_data(FULL_PATH_TO_DATASET)
        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        if labels_dict == None:
            self.labels_dict = self.get_label_to_ix()
        else:
            self.labels_dict = labels_dict
        
        self.maxlen = maxlen
        self.num_labels = len(self.labels_dict)

    def read_data(self, filename):
        data = pd.read_csv(filename, sep=";", names =['utterance','label'], header=None, dtype={'utterance':str, 'label': str} )
        return data

    def __len__(self):
        return len(self.df)

    def get_label_to_ix(self):
        label_to_ix = {}
        for label in self.df.label:
            if label not in label_to_ix:
                label_to_ix[label]=len(label_to_ix)
        return label_to_ix        

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'utterance']
        label = self.df.loc[index, 'label']
        label = self.labels_dict[label]

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

        tokens_ids_tensor = torch.tensor(token_ids) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()
        return tokens_ids_tensor, attn_mask, label
        