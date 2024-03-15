import os, sys

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir)

import random
from argparse import Namespace

import yaml
import torch
from datasets import load_from_disk, list_datasets
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from huggingface_hub import Repository
import wandb

from data_processing import load_preprocess_and_save, tokenize
from utils import set_seed, TempRandomSeed

with open(os.path.join(src_dir, "../config/artists.yaml"), 'r') as f:
    ARTISTS = yaml.load(f, Loader=yaml.FullLoader)
with open(os.path.join(src_dir, "../config/tokenizer.yaml"), 'r') as f:
    tokenizer_config = yaml.load(f, Loader=yaml.FullLoader)['config']
with open(os.path.join(src_dir, "../config/training.yaml"), 'r') as f:
    training_config = yaml.load(f, Loader=yaml.FullLoader)['config']
# Import the path configuration
with open(os.path.join(src_dir, "../config/paths.yaml"), 'r') as f:
    local_data_path = yaml.load(f, Loader=yaml.FullLoader)['local_paths']['data']
    local_chkpt_path = yaml.load(f, Loader=yaml.FullLoader)['local_paths']['checkpoints']
    hub_data_path = yaml.load(f, Loader=yaml.FullLoader)['hub_paths']['data']


def train(datasets_dict, tokenizer, train_config):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')
    for i, artist in enumerate(datasets_dict.keys()):

        # Generate a random string for the run name. Since we may have fixed the seed earlier,
        # generate a temporary random seed to ensure the run name is different each time.
        with TempRandomSeed(i):
            run_name = f"{artist}-{''.join([str(i) for i in random.sample(range(10), 6)])}"

        wandb.init(
            project="lyrics-dreamer",
            name=run_name,
            config={
                'artist': artist,
                'tokenizer': tokenizer_config,
                'train': train_config
            }
        )
        print(f"Training {artist}:")
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        training_args = TrainingArguments(
            output_dir=os.path.join(local_chkpt_path, artist),
            learning_rate=float(training_config['learning_rate']),
            weight_decay=float(training_config['weight_decay']),
            lr_scheduler_type=training_config['lr_scheduler_type'],
            evaluation_strategy='epoch',
            logging_strategy='steps',
            logging_dir=os.path.join(src_dir, '../logs'),
            save_strategy='epoch',
            save_steps=10,
            per_device_train_batch_size=int(training_config['batch_size']),
            push_to_hub=False,
            fp16=train_config['fp16'],
            logging_steps=10,
            # max_steps=int(training_config['n_training_steps']),
            num_train_epochs=int(training_config['n_epochs']),
            report_to=['wandb'],
            load_best_model_at_end=True,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
        # lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=int(training_config['n_warmup_steps']),
        #     num_training_steps=training_args.num_train_epochs * len(
        #         datasets_dict[artist]['train']
        #     ) // training_args.per_device_train_batch_size,
        #     # num_training_steps=training_args.max_steps,
        #     num_cycles=int(training_config['n_cosine_cycles']),
        # )
        lr_scheduler = None

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets_dict[artist]['train'],
            eval_dataset=datasets_dict[artist]['validation'],
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_config['early_stopping_patience'])],
        )
        trainer.train()
        if training_config['save_to_hub']:
            repo = Repository(
                os.path.join(local_chkpt_path, artist),
                clone_from=f"https://huggingface.co/lfrachon/lyrics-dreamer/checkpoints/{artist}"
            )
            repo.push_to_hub(os.path.join(local_chkpt_path, artist))
            repo.git_pull()
            model.save_pretrained(repo.local_dir)
            # tokenizer.save_pretrained(repo.local_dir)
            repo.git_add()
            repo.git_commit(f"Update {artist} model")
            repo.git_push()

        wandb.finish()


def add_artist_model(artist_name, force_retrain=False):
    available_artists = [dataset for dataset in list_datasets() if dataset.startswith('huggingartists')]
    available_artists = [artist.split('/')[-1] for artist in available_artists]

    if artist_name not in available_artists:
        print(f"Dataset `huggingartits/{artist_name}` not found in HuggingFace datasets. "
              f"Please select a different one.")
        choice = input("See available artists? [y/n]: ")
        if choice == 'y':
            print(available_artists)
        return

    if artist_name in ARTISTS and not force_retrain:
        print(f"The model for {artist_name} is already available.")
        return

    _ = load_preprocess_and_save(
        tokenize,
        block_size=tokenizer_config['block_size'],
        stride=tokenizer_config['stride'],
        artist_list=[artist_name],
        save_dir=local_data_path,
    )
    print(f"Dataset `huggingartists/{artist_name}` added to `data_tokenized`.")
    print(f"Training {artist_name}:")
    run_training_pipeline([artist_name])

    # Add the new artist to the list:
    if artist_name not in ARTISTS:
        ARTISTS.append(artist_name)
        with open(os.path.join(src_dir, "../config/artists.yaml"), 'w') as f:
            yaml.dump(ARTISTS, f)
        print(f"Artist {artist_name} added to `../config/artists.yaml`.")


def run_training_pipeline(artist_list: list = ARTISTS):
    set_seed(training_config['seed'])
    datasets = {
        artist: load_from_disk(os.path.join(local_data_path, artist)) for artist in artist_list
    }
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    train(datasets, tokenizer, training_config)


def main(args: Namespace):
    if args.add_1_artist:
        add_artist_model(args.artist, force_retrain=args.force_retrain)
    else:
        run_training_pipeline()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-1', '--add_1_artist', action='store_true', help='Add a new artist to the training pipeline.')
    parser.add_argument('-a', '--artist', type=str, default='queen', help='Name of the artist to add.')
    parser.add_argument('-f', '--force_retrain', action='store_true', help='Force retraining of the model even if it'
                                                                           'exists. Only use with `-1`.')
    args = parser.parse_args()
    main(args)
