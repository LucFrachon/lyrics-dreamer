import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TempRandomSeed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *args):
        random.setstate(self.state)


def pretty_format(generated_lyrics):
    # def wrap_text(text, max_width):
    #     original_lines = text.split('\n')
    #     wrapped_lines = []
    #
    #     for line in original_lines:
    #         words = line.split()
    #         current_line = []
    #
    #         for word in words:
    #             if len(" ".join(current_line + [word])) <= max_width:
    #                 current_line.append(word)
    #             else:
    #                 wrapped_lines.append(" ".join(current_line))
    #                 current_line = [word]
    #
    #         wrapped_lines.append(" ".join(current_line))  # Add the last line of the current group
    #     return "\n".join(wrapped_lines)


    if not isinstance(generated_lyrics, list):
        generated_lyrics = [generated_lyrics]
    out = '---\n'

    for i, lyric in enumerate(generated_lyrics):
        out += f"Lyrics {i + 1}:\n" if len(generated_lyrics) > 1 else "Lyrics:\n"
        out += lyric
        out += '\n---'
    return out
