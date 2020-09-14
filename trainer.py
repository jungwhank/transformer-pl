import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchtext
from torchtext.data import TabularDataset, BucketIterator
from model import Transformer
from utils import korean_tokenizer_load, english_tokenizer_load


class Transformer_pl(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super(Transformer_pl, self).__init__()
        self.hparams = hparams
        self.transformer = Transformer(self.hparams)

        self.sp_kor = korean_tokenizer_load()
        self.sp_eng = english_tokenizer_load()

    def forward(self, enc_inputs, dec_inputs):
        output_logits, *_ = self.transformer(enc_inputs, dec_inputs)
        return output_logits

    def cal_loss(self, tgt_hat, tgt_label):
        loss = F.cross_entropy(tgt_hat, tgt_label.contiguous().view(-1), ignore_index=self.hparams['padding_idx'])
        return loss

    def translate(self, input_sentence):
        self.eval()
        input_ids = self.sp_kor.EncodeAsIds(input_sentence)
        if len(input_ids) < self.hparams['max_seq_length']:
            input_ids = input_ids + [self.hparams['padding_idx']]*(self.hparams['max_seq_length'] - len(input_ids))
        input_ids = torch.tensor([input_ids])

        enc_outputs, _ = self.transformer.encode(input_ids)
        target_ids = torch.zeros(1, self.hparams['max_seq_length']).type_as(input_ids.data)
        next_token = self.sp_eng.bos_id()

        for i in range(0, self.hparams['max_seq_length']):
            target_ids[0][i] = next_token
            decoder_output, *_ = self.transformer.decode(target_ids, input_ids, enc_outputs)
            prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_token = prob.data[i].item()
            if next_token == self.sp_eng.eos_id():
                break

        output_sent = self.sp_eng.DecodeIds(target_ids[0].tolist())
        return output_sent

    # ---------------------
    # TRAINING AND EVALUATION
    # ---------------------
    def training_step(self, batch, batch_idx):
        src, tgt = batch.kor, batch.eng
        tgt_label = tgt[:, 1:]
        tgt_hat = self(src, tgt[:, :-1])
        loss = self.cal_loss(tgt_hat, tgt_label)
        train_ppl = math.exp(loss)
        tensorboard_logs = {'train_loss': loss, 'train_ppl': train_ppl}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        src, tgt = batch.kor, batch.eng
        tgt_label = tgt[:, 1:]
        tgt_hat = self(src, tgt[:, :-1])
        val_loss = self.cal_loss(tgt_hat, tgt_label)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_ppl = math.exp(val_loss)
        tensorboard_logs = {'val_loss': val_loss, 'val_ppl': val_ppl}
        print("")
        print("="*30)
        print(f"val_loss:{val_loss}")
        print("="*30)
        return {'val_loss': val_loss, 'val_ppl' : val_ppl, 'log': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=self.hparams['lr'])
        return [optimizer]

    def make_field(self):
        KOR = torchtext.data.Field(use_vocab=False,
                                   tokenize=self.sp_kor.EncodeAsIds,
                                   batch_first=True,
                                   fix_length=self.hparams['max_seq_length'],
                                   pad_token=self.sp_kor.pad_id())

        ENG = torchtext.data.Field(use_vocab=False,
                                   tokenize=self.sp_eng.EncodeAsIds,
                                   batch_first=True,
                                   fix_length=self.hparams['max_seq_length']+1,  # should +1 because of bos token for tgt label
                                   init_token=self.sp_eng.bos_id(),
                                   eos_token=self.sp_eng.eos_id(),
                                   pad_token=self.sp_eng.pad_id())
        return KOR, ENG

    def train_dataloader(self):
        KOR, ENG = self.make_field()
        train_data = TabularDataset(path="./data/train.tsv",
                                    format='tsv',
                                    skip_header=True,
                                    fields=[('kor', KOR), ('eng', ENG)])

        train_iter = BucketIterator(train_data,
                                    batch_size=self.hparams['batch_size'],
                                    sort_key=lambda x: len(x.kor),
                                    sort_within_batch=False)
        return train_iter

    def val_dataloader(self):
        KOR, ENG = self.make_field()
        valid_data = TabularDataset(path="./data/valid.tsv",
                                    format='tsv',
                                    skip_header=True,
                                    fields=[('kor', KOR), ('eng', ENG)])

        val_iter = BucketIterator(valid_data,
                                  batch_size=self.hparams['batch_size'],
                                  sort_key=lambda x: len(x.kor),
                                  sort_within_batch=False)
        return val_iter



