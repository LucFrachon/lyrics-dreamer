import os

from flask import Flask, render_template, request, jsonify
import yaml
import src.inference as inference

src_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['DEBUG'] = True

model_kwargs = dict(
    num_return_sequences=1,
    min_length=100,
    max_length=200,
    temperature=1.41,
    top_p=0.95,
    top_k=0,
    repetition_penalty=1.1,
    early_stopping=True,
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


@app.route('/')
def index():
    artists = load_artists()
    return render_template('index.html', artists=artists)

@app.route('/generate_lyrics', methods=['POST'])
def generate_lyrics():
    artist = request.form.get('artist')
    prompt = request.form.get('prompt')
    model, tokenizer = inference.initialise_model_for_inference(artist)
    lyrics = inference.generate_lyrics(
        model,
        tokenizer,
        prompt,
        **model_kwargs,
    )
    lyrics = inference.pretty_format(lyrics)
    return jsonify({"lyrics": lyrics})


if __name__ == '__main__':
    app.run(debug=True)
