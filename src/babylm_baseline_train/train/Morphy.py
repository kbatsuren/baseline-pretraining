from transformers import AutoTokenizer
import json

class Morphy:
    def __init__(self, vocab_file, merges_file, unimorph_file):
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.encoder["<s>"] = len(self.encoder)
        self.encoder["<pad>"] = len(self.encoder)
        self.encoder["</s>"] = len(self.encoder)
        self.encoder["<unk>"] = len(self.encoder)
        self.encoder["<mask>"] = len(self.encoder)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.surf_merges = dict()
        self.morp_merges = dict()
        with open(merges_file, encoding="utf-8") as merges_handle:
            for line in merges_handle:
                parts = line.rstrip("\n").split(" ")
                if len(parts) < 4:
                  continue
                if parts[2] == 'm' or parts[2]=='b':
                  self.morp_merges[parts[0],parts[1]] = parts[3]
                if parts[2] == 's' or parts[2] =='b':
                  self.surf_merges[parts[0],parts[1]] = parts[3]
        # self.bpe_merges = {tuple(merge.split('\t')) : "".join(merge.split()) for merge in merges}
        # with open(morphy_merges_file, encoding="utf-8") as merges_handle:
        #     morphy_merges = merges_handle.read().split("\n")[0:-1]
        # self.morphy_merges = {tuple(merge.split('\t')[:2]) : merge.split('\t')[2] for merge in morphy_merges}
        data = {}
        with open(unimorph_file, encoding='utf-8') as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if fields[0][0].islower():
                    data[fields[0]] = fields[1].lower().split(' @@')
        self.segmentations=data
        self.pre_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def initialize_word(self, word):
        split = []
        if word in self.encoder:
            return [word]
        if word.startswith('Ġ') is False:
            if (word in self.segmentations) is False:
                split = [c for c in word]
            else:
                morpheme_split = self.segmentations[word]
                for morpheme in morpheme_split:
                    if morpheme in self.encoder:
                      split.append('Ṡ')
                      split.append(morpheme)
                    else:
                      split.append('Ṡ')
                      for c in morpheme:
                        split.append(c)
                split.append('Ṡ')
        elif (word[1:] in self.segmentations) is False:
            split = [c for c in word[1:]]
            split[0] = 'Ġ'+split[0]
        else:
            split = []
            morpheme_split = [] + self.segmentations[word[1:]]
            #morpheme_split[0] = 'Ġ'+morpheme_split[0]
            for morpheme in morpheme_split:
                if morpheme in self.encoder:
                  split.append('Ṡ')
                  split.append(morpheme)
                else:
                  split.append('Ṡ')
                  for c in morpheme:
                      split.append(c)
            split.append('Ṡ')
            split[1] = 'Ġ'+split[1]
        return split

    def tokenize(self, text):
        pre_tokenize_result = self.pre_tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits = [self.initialize_word(word) for word in pre_tokenized_text]
        unimorph_exists = [word in self.segmentations or (word[0]=='Ġ' and word[1:] in self.segmentations) for word in pre_tokenized_text]
        for pair, merge in self.surf_merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
        #print(splits)
        for pair, merge in self.morp_merges.items():
            for idx, split in enumerate(splits):
                if unimorph_exists[idx] == False:
                    continue
                #print('orloo')
                i = 1
                while i < len(split) - 3:
                    if split[i] == pair[0] and split[i + 2] == pair[1]:
                        if split[i-1] == 'Ṡ' and split[i+1] == 'Ṡ' and split[i+3] == 'Ṡ':
                          split = split[:i] + [merge] + split[i + 3 :]
                    else:
                        i += 1
                splits[idx] = split

        for idx, split in enumerate(splits):
          if unimorph_exists[idx] == False:
            continue
          sp = []
          for subword in split:
            if subword!='Ṡ':
              sp.append(subword)
          splits[idx] = sp
        return ['<s>'] + sum(splits, []) + ['</s>']

    def __call__(self, examples):
        input = []
        mask = []
        for example in examples:
          split = self.tokenize(example)
          split_ids = []
          for subword in split:
            split_ids.append(self.encoder[subword])
          input.append(split_ids)
          mask.append([1] * len(split))
        return {'input_ids': input,
               'attention_mask': mask}
  