########### Here we create dataset in form of tensor to feed our transformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src,tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # create tokens
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)
        
    ## define length method of the dataset
    def __len__(self):
        return len(self.ds)
    ## get item method
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        ## each text into tokens then into input IDs
        # tokenizer will split sentence into word then map to corresponding vocabulary in one pass only
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # paddding token- fill empty space into sentence to reach seq length
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1
        
        # padding number should never become negative
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens <0:
            raise ValueError('Sentence is too long')
        
        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        # add SOS to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        # create label - add EOS to the label --> what we expect as output from the decoder
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        ## debugging
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input, #(seq_len)
            "decoder_input" : decoder_input, #(seq_len)
            # we do not want this padding tokens to be seen by self attention mechanism, so use mask
            "encoder_mask": (encoder_input !=self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_Len)
            "decoder_mask": (decoder_input !=self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text" : src_text,
            "tgt_text": tgt_text
                
        }
        
# define causal mask - one word can watch itself (diagonal elmements), others are hidden
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)  # triu-> returns all values above diagonal
    return mask ==0