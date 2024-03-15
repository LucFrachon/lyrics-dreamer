import os, sys

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir)

import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import set_seed, pretty_format

with open(os.path.join(src_dir, "../config/inference.yaml"), 'r') as f:
    inference_config = yaml.load(f, Loader=yaml.FullLoader)['config']
if inference_config['mode'] == 'local':
    with open(os.path.join(src_dir, "../config/paths.yaml"), 'r') as f:
        chkpt_path = yaml.load(f, Loader=yaml.FullLoader)['local_paths']['checkpoints']
elif inference_config['mode'] == 's3':
    with open(os.path.join(src_dir, "../config/paths.yaml"), 'r') as f:
        chkpt_path = yaml.load(f, Loader=yaml.FullLoader)['s3_paths']['checkpoints']


def initialise_model_for_inference(artist_id: str):
    if inference_config['mode'] == 'local':
        checkpoint_dir = os.path.join(chkpt_path, artist_id)
        # Get the most recent checkpoint
        step_nums = [int(f.split('-')[1]) for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
        step_nums.sort()
        latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint-{step_nums[-1]}")
    elif inference_config['mode'] == 's3':
        latest_checkpoint = f"{chkpt_path}{artist_id}/checkpoint"
    else:
        return ValueError, "Only `local` and `s3` are valid modes."

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)

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
    if inference_config['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    encoded_prompt = tokenizer(prompt, return_tensors='pt')
    encoded_prompt = {k: v.to(device) for k, v in encoded_prompt.items()}
    model.to(device)
    output = model.generate(
        input_ids=encoded_prompt['input_ids'],
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


def main(args):
    set_seed(args.seed)
    model, tokenizer = initialise_model_for_inference(args.artist)
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
    print(pretty_format(generated_lyrics))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--artist', '-a', type=str, required=True)
    parser.add_argument('--prompt', '-p', type=str, required=True)
    parser.add_argument('--num_return_sequences', '-n', type=int, default=1)
    parser.add_argument('--min_length', '-i', type=int, default=100)
    parser.add_argument('--max_length', '-x', type=int, default=200)
    parser.add_argument('--temperature', '-t', type=float, default=1.2)
    parser.add_argument('--top_p', '-o', type=float, default=0.95)
    parser.add_argument('--top_k', '-k', type=int, default=0)
    parser.add_argument('--repetition_penalty', '-r', type=float, default=1.0)
    parser.add_argument('--early_stopping', '-e', action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=42)
    args = parser.parse_args()

    main(args)
