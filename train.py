#### Here we will download our dataset from huggingface and make training pipeline
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

import warnings
import pathlib
from pathlib import Path



def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang): # config file, ds is dataset, lang is langguage
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer  =Tokenizer(WordLevel(unk_token='[UNK]')) # unk - unkown
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[PAD]", "[EOS]"], min_frequency=2) # unkown, padding, start of sentence, end of sentence
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        
### Load the dataset

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    
    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config=['lang_tgt'])
    
    # as we have only train split from hugging face, we can do further split ourselves
    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # random_split is from torch, split datasets by given sizes
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # we also want to see max length of sentece in src and tgt for each splits
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True) # 1 becausem we want to validate sentence one by one
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


#### Build model

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


## Train loop
def train_model(config):
    # define device
    device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')
    print(f'Using device {device}')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # load dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # tensorboard --> visualize loss
    writer = SummaryWriter(config['experiment_name'])
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # when model crash, it restore model state again
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(['optimizer_state_dict'])
        global_step = state['global_step']
    
    # loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1 ).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # ( Batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # ( batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # ( B, 1, seq_len, seq_len)
            
            # run the tensor through transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # ( B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) #( B, seq_len, tgt_vocab_size
            
            # now we have model output, now compare it with labels
            label = batch['label'].to(device) # (B, seq_len)
            
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            # log the loss on tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the weights
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
        ## Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            # it is nice to record optimizer's progress with model
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
## code to run the model
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
    
            