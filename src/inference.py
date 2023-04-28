import os

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import set_seed

src_dir = os.path.dirname(os.path.abspath(__file__))


def initialise_model(artist_name: str):
    checkpoint_dir = os.path.join(src_dir, f"../checkpoints/{artist_name}")
    # Get the most recent checkpoint
    step_nums = [int(f.split('-')[1]) for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
    step_nums.sort()
    latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint-{step_nums[-1]}")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
    model.eval()

    return model, tokenizer


def generate_lyrics(
        model,
        tokenizer,
        prompt,
        num_return_sequences,
        min_length,
        max_length,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        early_stopping,
):
    encoded_prompt = tokenizer(prompt, return_tensors='pt')
    output = model.generate(
        input_ids =encoded_prompt['input_ids'],
        attention_mask=encoded_prompt['attention_mask'],
        do_sample=True,
        min_length=min_length,
        max_length=max_length,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        early_stopping=early_stopping,
        num_return_sequences=num_return_sequences,
    )
    generated_lyrics = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]

    return generated_lyrics


def pretty_print(generated_lyrics):
    print('---')
    for i, lyric in enumerate(generated_lyrics):
        print(f"Lyric {i + 1}:")
        print(lyric)
        print('---')

def main(args):
    set_seed(args.seed)
    model, tokenizer = initialise_model(args.artist)
    generated_lyrics = generate_lyrics(
        model,
        tokenizer,
        args.prompt,
        num_return_sequences=args.num_return_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        early_stopping=args.early_stopping,
    )
    pretty_print(generated_lyrics)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--artist', '-a', type=str, required=True)
    parser.add_argument('--prompt', '-p', type=str, required=True)
    parser.add_argument('--num_return_sequences', '-n', type=int, default=1)
    parser.add_argument('--min_length', '-i', type=int, default=100)
    parser.add_argument('--max_length', '-x', type=int, default=200)
    parser.add_argument('--temperature', '-t', type=float, default=1.)
    parser.add_argument('--top_p', '-o', type=float, default=0.98)
    parser.add_argument('--top_k', '-k', type=int, default=0)
    parser.add_argument('--repetition_penalty', '-r', type=float, default=1.0)
    parser.add_argument('--early_stopping', '-e', action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=42)
    args = parser.parse_args()

    main(args)
