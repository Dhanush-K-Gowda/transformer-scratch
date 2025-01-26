from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer  # For loading tokenizers from JSON

import warnings
from tqdm import tqdm
from pathlib import Path

# Hugging Face datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# CHANGE: Added Hugging Face Hub integration
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# CHANGE: Log in to Hugging Face Hub
login(token)  # Replace with your Hugging Face token

# WandB for experiment tracking (optional)
import wandb
# Torchmetrics for evaluation metrics
import torchmetrics
from huggingface_hub import HfApi
from google.colab import drive


def mount_google_drive():
    """Mount Google Drive and return checkpoint path"""
    drive.mount('/content/drive')
    checkpoint_dir = "/content/drive/My Drive/transformer_checkpoints/"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

def save_checkpoint(model, optimizer, tokenizer_src, tokenizer_tgt, epoch, checkpoint_dir):
    """Save checkpoint to Google Drive"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer_src': tokenizer_src.to_str(),
        'tokenizer_tgt': tokenizer_tgt.to_str(),
        'config': config
    }
    torch.save(checkpoint, f"{checkpoint_dir}checkpoint_epoch_{epoch:02d}.pt")

def load_latest_checkpoint(checkpoint_dir, device):
    """Load latest checkpoint from Google Drive"""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None, None, None, None, 0

    # Find latest epoch
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    checkpoint_path = f"{checkpoint_dir}checkpoint_epoch_{latest_epoch:02d}.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return (
        checkpoint['model_state_dict'],
        checkpoint['optimizer_state_dict'],
        checkpoint['tokenizer_src'],
        checkpoint['tokenizer_tgt'],
        checkpoint['epoch'] + 1  # Start from next epoch
    )



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break


    # Evaluate the character error rate
    # Compute the char error rate
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))  # CHANGE: Explicitly save the tokenizer
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# def get_or_build_tokenizer(config, ds, lang):
#   tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
#   return tokenizer



def get_ds(config):
    # Load the dataset (single split: "train")
    ds_raw = load_dataset(
        "ai4bharat/samanantar",
        f"{config['lang_tgt']}",
        split="train"  # Returns a Dataset, not DatasetDict
    )

    # Select a subset (e.g., 20,000 samples)
    ds_raw = ds_raw.select(range(2000))

    # Rename columns to match OPUS format
    ds_raw = ds_raw.map(
        lambda x: {"translation": {config['lang_src']: x["src"], config['lang_tgt']: x["tgt"]}},
        remove_columns=["src", "tgt", "idx"]
    )

    # Split into train/validation (80-20)
    ds_split = ds_raw.train_test_split(test_size=0.2, seed=42)
    train_ds_raw = ds_split["train"]  # Access via split keys
    val_ds_raw = ds_split["test"]

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw, config['lang_tgt'])

    # Initialize datasets
    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len']
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len']
    )

    return train_ds, val_ds, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model




def train_model(config):
    # Mount Google Drive and set up checkpoint directory
    checkpoint_dir = "/content/drive/My Drive/transformer/transformer_checkpoints/"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Try to load latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoints:
        # Find latest epoch
        epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
        latest_epoch = max(epochs)
        checkpoint_path = f"{checkpoint_dir}checkpoint_epoch_{latest_epoch:02d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load state from checkpoint
        tokenizer_src = checkpoint['tokenizer_src']
        tokenizer_tgt = checkpoint['tokenizer_tgt']
        # Recreate DataLoaders with existing tokenizers
        train_dataloader, val_dataloader, _, _ = get_ds(config, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt)

        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        initial_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        print(f"Resuming training from epoch {latest_epoch}")
    else:
        # Initialize from scratch
        print("No checkpoints found. Starting new training session.")
        initial_epoch = 0
        global_step = 0
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Rest of the training loop remains unchanged
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Calculate loss
            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Logging
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})
            global_step += 1

        # Validation and checkpointing
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                      lambda msg: batch_iterator.write(msg), global_step)

        # Save checkpoint to Google Drive
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tokenizer_src': tokenizer_src,
            'tokenizer_tgt': tokenizer_tgt,
            'config': config
        }
        torch.save(checkpoint, f"{checkpoint_dir}checkpoint_epoch_{epoch:02d}.pt")
        print(f"Saved checkpoint for epoch {epoch} to Google Drive")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config['num_epochs'] = 20

    # Mount Google Drive
    # from google.colab import drive
    # drive.mount('/content/drive')

    # Initialize wandb
    wandb.init(project="pytorch-transformer", config=config)

    train_model(config)
