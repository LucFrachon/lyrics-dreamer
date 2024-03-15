import os
import random

from flask import Flask, render_template, request, jsonify, session, send_from_directory
import yaml

from src import inference, utils

src_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = "280548"

model_kwargs = dict(
    num_return_sequences=1,
    min_length=100,
    max_length=200,
    temperature=1.2,
    top_p=0.95,
    top_k=0,
    repetition_penalty=1.0,
    early_stopping=False,
)


def load_artists(artist_list: list = None):
    """
    Load artists from artists.yaml or from a list of artist ids. An artist id is the artist name in lower case with
    spaces replaced by hyphens, as per the dataset's convention.
    :param artist_list: List (optional), mainly for debugging purposes. Pass a list of artists ids to clean up.
    :return: Dict, Key = artist id, Value = formatted artist name.
    """

    def cleanup_names(name: str):
        name_clean = name.replace('-', ' ').replace('_', ' ')
        # upper case each first letter of a word
        name_clean = ' '.join([word.capitalize() for word in name_clean.split(' ')])
        return name_clean

    artists_dict = {}
    if artist_list is None:
        with open(os.path.join(src_dir, "config/artists.yaml"), 'r') as f:
            raw_artists = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raw_artists = artist_list

    clean_artists = map(cleanup_names, raw_artists)
    artists_dict = dict(zip(raw_artists, clean_artists))

    return artists_dict


ARTISTS_DICT = load_artists()
with open(os.path.join(src_dir, "config/game_prompts.yaml"), 'r') as f:
    PROMPTS = yaml.load(f, Loader=yaml.FullLoader)

def generate_lyrics_for_display(artist_id, prompt):
    model, tokenizer = inference.initialise_model_for_inference(artist_id)
    lyrics = inference.generate_lyrics(
        model,
        tokenizer,
        prompt,
        **model_kwargs,
    )
    lyrics = utils.pretty_format(lyrics)
    return lyrics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/lyrics_generator', methods=['GET', 'POST'])
def lyrics_generator():
    if request.method == 'POST':
        artist_id = request.form.get('artist')
        prompt = request.form.get('prompt')
        lyrics = generate_lyrics_for_display(artist_id, prompt)
        return jsonify(lyrics=lyrics)
    else:
        return render_template('lyrics_generator.html', artists=ARTISTS_DICT)

@app.route('/game', methods=['GET'])
def game():
    session['correct_guesses'] = 0
    session['total_guesses'] = 0
    session['score'] = '0'
    session['rounds'] = 0
    return render_template("game.html", artists=ARTISTS_DICT)

@app.route('/generate_game_lyrics', methods=['POST'])
def generate_game_lyrics():
    artist_id = random.choice(list(ARTISTS_DICT.keys()))
    prompt = random.choice(PROMPTS)
    session['artist'] = artist_id
    print(artist_id, prompt)
    lyrics = generate_lyrics_for_display(artist_id, prompt)
    print(lyrics)
    return jsonify(lyrics=lyrics)

@app.route('/submit_guess', methods=['POST'])
def submit_guess():
    guess = request.form.get('guess')
    artist_id = session.get('artist')

    if artist_id and guess == artist_id:
        result = "Correct!"
        session['correct_guesses'] += 1
    else:
        result = "Incorrect, it was " + ARTISTS_DICT[artist_id] + "."

    session['rounds'] += 1
    session['score'] = f"{100. * session['correct_guesses'] / session['rounds']:.1f}"

    return jsonify(result=result, score=session['score'], rounds=session['rounds'])


@app.route('/favicon')
def favicon():
    return send_from_directory('static', 'assets/favicon.ico')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
