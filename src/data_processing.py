import os
from functools import partial

import yaml

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict

src_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Add the EOS token as PAD token to avoid warnings and allow PyTorch to create tensors
tokenizer.pad_token = tokenizer.eos_token
# Import the artist names
with open(os.path.join(src_dir, "../config/artists.yaml"), 'r') as f:
    ARTISTS = yaml.load(f, Loader=yaml.FullLoader)
# Import the tokenizer configuration
with open(os.path.join(src_dir, "../config/tokenizer.yaml"), 'r') as f:
    tokenizer_config = yaml.load(f, Loader=yaml.FullLoader)['config']


def get_and_clean_data(
        artist: str,
        raw_data_root: str = None,
        clean_data_root: str = None,
        test_dataset: DatasetDict = None
):
    if test_dataset is not None:
        # For testing and debugging purposes only
        ds = test_dataset
    else:
        ds = load_dataset(f"huggingartists/{artist}")
    if raw_data_root is not None:
        os.makedirs(raw_data_root, exist_ok=True)
        ds.save_to_disk(os.path.join(raw_data_root, artist))
    # Remove song listings
    ds['train'] = ds['train'].filter(lambda x: not x['text'].startswith('1'))
    # Remote empty sentences
    ds['train'] = ds['train'].filter(lambda x: len(x['text']) > 0)
    if clean_data_root is not None:
        os.makedirs(clean_data_root, exist_ok=True)
        ds.save_to_disk(os.path.join(clean_data_root, artist))
    return ds


def split_datasets(ds=None, valid_size=0.1, load_dir: str = None, save_dir: str = None, non_random: bool = False):
    if ds is None:
        if load_dir is None:
            raise ValueError("Either ds or load_dir must be provided.")
        ds = load_from_disk(load_dir)
    ds_split = ds['train'].train_test_split(test_size=valid_size, shuffle=not non_random)
    ds_split['validation'] = ds_split.pop('test')
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        ds_split.save_to_disk(save_dir)
    return ds_split


def tokenize(examples: dict, block_size: int = 128, stride: int = 128):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=block_size,
        stride=stride,
        padding=False,
        return_overflowing_tokens=True,
    )


def load_preprocess_and_save(
        tokenizer_function: callable,
        block_size: int = 256,
        stride: int = 128,
        valid_size: float = 0.1,
        save_dir: str = os.path.join(src_dir, '../data_tokenized'),
        save_raw: bool = False,
        save_clean: bool = False,
        save_split: bool = False,
        artist_list: list = ARTISTS,
):
    tokenized_datasets = {}
    os.makedirs(save_dir, exist_ok=True)

    for artist in artist_list:
        print(f"Processing {artist}")
        ds = get_and_clean_data(
            artist,
            raw_data_root=os.path.join(src_dir, '../data_raw') if save_raw else None,
            clean_data_root=os.path.join(src_dir, '../data_clean') if save_clean else None,

        )
        ds = split_datasets(
            ds,
            valid_size=valid_size,
            save_dir=os.path.join(src_dir, f'../data_split/{artist}') if save_split else None,
        )
        tokenizer_partial = partial(tokenizer_function, block_size=block_size, stride=stride)
        tokenized_datasets[artist] = ds.map(tokenizer_partial, batched=True, remove_columns=['text'])
        tokenized_datasets[artist].remove_columns(['overflow_to_sample_mapping'])
        tokenized_datasets[artist].save_to_disk(os.path.join(save_dir, artist))
    return tokenized_datasets


if __name__ == '__main__':
    _ = load_preprocess_and_save(
        tokenize,
        block_size=tokenizer_config['block_size'],
        stride=tokenizer_config['stride'],
        save_split=True,
    )
