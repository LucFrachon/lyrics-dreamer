"""
Pick an artist at random from artists.yaml, generate lyrics, and ask the user to guess whose style the lyrics are in.
This script is only for using on the command line, not for the web app.
"""
import os, sys

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir)

import random

import yaml

from inference import initialise_model_for_inference, generate_lyrics
from utils import set_seed, pretty_format

with open(os.path.join(src_dir, "../config/artists.yaml"), 'r') as f:
    ARTISTS = yaml.load(f, Loader=yaml.FullLoader)
with open(os.path.join(src_dir, "../config/tokenizer.yaml"), 'r') as f:
    tokenizer_config = yaml.load(f, Loader=yaml.FullLoader)['config']
with open(os.path.join(src_dir, "../config/training.yaml"), 'r') as f:
    training_config = yaml.load(f, Loader=yaml.FullLoader)['config']
with open(os.path.join(src_dir, "../config/game_prompts.yaml"), 'r') as f:
    PROMPTS = yaml.load(f, Loader=yaml.FullLoader)


def main(args):
    if args.seed:
        set_seed(args.seed)
    artist = random.choice(ARTISTS)
    prompt = random.choice(PROMPTS)
    model, tokenizer = initialise_model_for_inference(artist)

    print(f"Generating lyrics...")
    lyrics = generate_lyrics(
        model,
        tokenizer,
        prompt,
        num_return_sequences=1,
        min_length=args.min_length,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        early_stopping=args.early_stopping,
    )
    lyrics = pretty_format(lyrics)
    print(lyrics)
    print(f"Guess whose lyrics these are: {', '.join(ARTISTS)}")
    guess = input(">>> ")
    if guess == artist:
        print("Correct!")
    else:
        print(f"Wrong, these lyrics are in the style of {artist}.")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--min_length", "-n", type=int, default=100)
    parser.add_argument("--max_length", "-x", type=int, default=200)
    parser.add_argument("--temperature", "-t", type=float, default=1.2)
    parser.add_argument("--top_p", "-o", type=float, default=0.95)
    parser.add_argument("--top_k", "-k", type=int, default=0)
    parser.add_argument("--repetition_penalty", "-r", type=float, default=1.0)
    parser.add_argument("--early_stopping", "-e", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=None)
    args = parser.parse_args()

    main(args)
