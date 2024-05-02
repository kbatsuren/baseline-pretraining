import os
import pdb
import setuptools
import torch

from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
from Morphy import Morphy
from babylm_baseline_train.env_vars import ROOT_DIR


def get_gpt2_tokenizer_func(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return tokenizer


def get_roberta_tokenizer_func(model_name="roberta-base"):
    voc_file = os.path.join(ROOT_DIR, 'tokenizers', "morphy-vocab.16000.m100.json")
    merge_file = os.path.join(ROOT_DIR, 'tokenizers', "morphy-merges.16000.m100.txt")
    um_file = os.path.join(ROOT_DIR, 'tokenizers', "eng.word.full.230613.r6.tsv")
    tokenizer = Morphy(voc_file, merge_file, um_file)

    print(tokenizer.tokenize("Our negotiation regarding women and children take paramount position within the discussions, but obviously we are moving towards civilian men being released"))

    return tokenizer

def get_tokenizer_func(opt_model_size='125m'):
    model_name = f"facebook/opt-{opt_model_size}"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_bos_token = False
    tokenizer.add_special_tokens(
            {
                'bos_token': '<s>', 
                'unk_token': '<unk>',
                'additional_special_tokens': [
                    '<image>', '</c>', 
                    '<PERSON>', # C-12M for person names
                    ]
            })
    return tokenizer
