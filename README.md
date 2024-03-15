# Lyrics Dreamer


An NLP application that learns to dream up lyrics in the style of various indie artists or groups.

You can either provide the start of a sentence for the model to complete, or play a game of "recognise the lyrical style".

## Local installation

Setting up a server and storage for the data and models costs money so I stopped. You can still run the application locally, here's how:

- I stored the checkpoints using Git LFS, so you need to install it first. For instance on Debian and Ubuntu, call `sudo apt-get install git-lfs`. See the complete instructions [here](https://github.com/git-lfs/git-lfs/wiki/Installation#installing).
- Clone the repo. This will take a few minutes due to the large files.
- `python3 -m venv .venv`
- `source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows)
- Install PyTorch according to the official instructions [here](https://pytorch.org/get-started/locally/). The code was developed and tested on PyTorch 2.0.0; if you have problems with a more recent version, try downgrading to that one ([see here](https://pytorch.org/get-started/previous-versions/)). You don't need the GPU version to run the application.
- `pip install -r requirements`

## Running

- Execute `flask run` 
- Enter the following URL in a browser: `http://127.0.0.1:5000`
- Have fun!


## Comments

The underlying models are GPT-2s fine-tuned on each artist's collection of lyrics, which I found on Aleksey Korshuk's repo [here](https://huggingface.co/datasets?search=huggingartists).

The generated lyrics don't always make sense and sometimes overfit to existing lyrics, but you can definitely tell the difference between Bob Dylan and Florence+The Machine!
