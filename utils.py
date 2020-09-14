import json
import sentencepiece as spm

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

def korean_tokenizer_load():
    sp_kor = spm.SentencePieceProcessor()
    sp_kor.Load('{}.model'.format("./tokenizer/kor"))
    return sp_kor

def english_tokenizer_load():
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load('{}.model'.format("./tokenizer/eng"))
    return sp_eng
