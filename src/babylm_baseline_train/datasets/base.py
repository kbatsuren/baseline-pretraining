from transformers import AutoTokenizer
import torch
from abc import ABC, abstractmethod
from transformers import GPT2Tokenizer
import ipdb
from tqdm import tqdm

from .utils import Group_Texts
from ..env_vars import ROOT_DIR, ROOT_DIR_FREQ
import os

TOKEN_SAVE_FOLDER = os.environ.get(
        'BABYLM_TOKEN_SAVE_FOLDER',
        os.path.join(
            ROOT_DIR, 'tokens/'))

class BaseGroupDataset(ABC):
    def __init__(self, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        words = set()
        spaces = set()
        for i in range(len(tokenizer)):
            subword = tokenizer.decode(i)
            if subword.startswith(' '):
                if subword[1:].isalpha():
                    spaces.add(i)
            elif subword.isalpha():
                words.add(i)
        replaces = dict()
        indx = -1
        with open(TOKEN_SAVE_FOLDER+'/roberta.all.replacements.v3.tsv', encoding='utf-8') as f:
          for line in f:
            indx+=1
            if indx == 0:
              continue
            fields = line.rstrip("\n").split("\t")
            replaces[fields[2].replace(' ','_')]=list(map(int, fields[3].split(' '))) 
        self.suffix_subwords = words 
        self.head_subwords = spaces
        self.replaces = replaces
        self.rep_tok_num = 0
        self.rep_word_num = 0
        self.total_tok_num = 0

    def prepare_tokenizer(self):
        if self.tokenizer is None:
            #self.tokenizer = AutoTokenizer.from_pretrained(
            #        "gpt2", fast=False)
            model_name = f"facebook/opt-125m"
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    @abstractmethod
    def get_dataset(self):
        pass

    def tokenize_function(self, examples):
        outputs = self.tokenizer(examples['text'])
        f_token = open(TOKEN_SAVE_FOLDER+"replace.records.roberta-base.v3.tsv", "a", encoding = "utf-8")
        indd = 0

        rets = {'input_ids':[], 'attention_mask': []}
        

        for i in range(len(outputs['input_ids'])):
          candidate = []
          result = []
          focus = [1]
          self.total_tok_num += len(outputs['input_ids'][i])
          for inp_id in outputs['input_ids'][i]:
            if len(candidate) == 0:
              if inp_id in self.head_subwords:
                candidate.append(inp_id)
              else:
                result.append(inp_id)
                continue
            elif inp_id in self.suffix_subwords:
              candidate.append(inp_id)
              continue
            else:
              rep_str = '_'.join(map(str, candidate))
              if rep_str in self.replaces:
                result+= list(map(int, self.replaces[rep_str])) 
                self.rep_tok_num += len(candidate)
                self.rep_word_num += 1
                f_token.write('replaced tokens: '+str(self.rep_tok_num)+'\treplaced words: '+str(self.rep_word_num)+'\ttotal_tokens: '+str(self.total_tok_num)+'\tchanged:'+rep_str+'\n')
              else:
                result+=candidate
              candidate = []
              result.append(inp_id)
          if len(candidate)!=0:
            rep_str = '_'.join(map(str, candidate))
            if rep_str in self.replaces:
              result+= list(map(int, self.replaces[rep_str])) 
              self.rep_tok_num += len(candidate)
              self.rep_word_num += 1
              f_token.write('replaced tokens: '+str(self.rep_tok_num)+'\treplaced words: '+str(self.rep_word_num)+'\ttotal_tokens: '+str(self.total_tok_num)+'\tchanged:'+rep_str+'\n')
              #print('replaced ', rep_tok_num, rep_word_num)
            else:
              result+=candidate
          focus *= len(result)
          # print(tokenizer.decode(outputs['input_ids'][i]))
          # print(tokenizer.decode(result), )
          # print(len(outputs['input_ids'][i]),len(result), len(outputs['attention_mask'][i]), len(result))
          # print(outputs['input_ids'][i])
          # print(outputs['attention_mask'][i])
          # print(result)
          # print(focus)
          rets['input_ids'].append(result)
          rets['attention_mask'].append(focus)
        f_token.close()

        # for example in examples['text']:
        #     indd+=1
        #     f_token.write('=================================\n')
        #     f_token.write(str(indd)+'\n')
        #     f_token.write('=================================\n')
        #     f_token.write(example+'\n')
        #     f_token.write('=================================\n')
        # f_token.close()
        return rets

    def get_group_dataset(self, just_dataset=False):
        self.prepare_tokenizer()
        self.get_dataset()
        if just_dataset == True:
            return self.dataset
        elif just_dataset == 'self':
            return self

        tokenized_datasets = self.dataset.map(
                self.tokenize_function, batched=True, 
                remove_columns=["text"])
        group_text_default = Group_Texts(
                tokenized_datasets, self.tokenizer, 
                seq_len=self.seq_len)

        grouped_dataset_default = group_text_default.group_texts()
        return grouped_dataset_default

    def count_num_of_words(self):
        import re
        import inflect
        import nltk.data
        from tqdm import tqdm

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        num = 0
        for data in tqdm(self.dataset):
            sents = tokenizer.tokenize(data['text'])
            for sent in sents:
                tokens = re.findall('\w+', sent)
                num += len(tokens)
        return num

    def count_num_of_tks(self):
        num_of_tks = 0
        for line in tqdm(self.dataset):
            txt_in_tks = self.tokenize_function(line)
            num_of_tks += len(txt_in_tks.input_ids)
        return num_of_tks
