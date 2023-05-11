# Tests for the data_processing.py script

import os
import unittest
from functools import partial

from datasets import load_from_disk, DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer

from src.data_processing import get_and_clean_data, split_datasets, tokenize, load_preprocess_and_save

# Set up the test data
TEST_DATA_ROOT = './test_data'
TEST_DATA_RAW_ROOT = os.path.join(TEST_DATA_ROOT, 'raw')
TEST_DATA_CLEAN_ROOT = os.path.join(TEST_DATA_ROOT, 'clean')
TEST_DATA_SPLIT_ROOT = os.path.join(TEST_DATA_ROOT, 'split')
TEST_DATA_TOKENIZED_ROOT = os.path.join(TEST_DATA_ROOT, 'tokenized')

TEST_ARTISTS = ['artist1', 'artist2', 'artist3']

TEST_DATA_RAW = {
    'artist1': DatasetDict({
        'train': Dataset.from_dict({
            'text': ['artist1 song1', 'artist1 song2', '',
                     'artist1 song4', 'artist1 song5', 'artist1 song6']
            }),
        }),
    'artist2': DatasetDict({
        'train': Dataset.from_dict({
            'text': ['artist2 song1', 'artist2 song2', 'artist2 song3',
                    '1. artist2 song4', 'artist2 song5', 'artist2 song6'],
        }),
    }),
    'artist3': DatasetDict({
        'train': Dataset.from_dict({
            'text': ['artist3 song1', '1 artist3 song2', 'artist3 song3',
                    'artist3 song4', 'artist3 song5', 'artist3 song6'],
        }),
    }),
}

TEST_DATA_CLEAN = {
    'artist1': {
        'train': Dataset.from_dict({
            'text': ['artist1 song1', 'artist1 song2', 'artist1 song4', 'artist1 song5', 'artist1 song6'],
        }),
    },
    'artist2': {
        'train': Dataset.from_dict({
            'text': ['artist2 song1', 'artist2 song2', 'artist2 song3', 'artist2 song5', 'artist2 song6'],
        }),
    },
    'artist3': {
        'train': Dataset.from_dict({
            'text': ['artist3 song1',  'artist3 song3', 'artist3 song4', 'artist3 song5', 'artist3 song6'],
        }),
    },
}

TEST_DATA_SPLIT = {
    'artist1': {
        'train': Dataset.from_dict({
            'text': ['artist1 song1', 'artist1 song2', 'artist1 song4'],
        }),
        'validation': Dataset.from_dict({
            'text': ['artist1 song5', 'artist1 song6'],
        }),
    },
    'artist2': {
        'train': Dataset.from_dict({
            'text': ['artist2 song1', 'artist2 song2', 'artist2 song3'],
        }),
        'validation': Dataset.from_dict({
            'text': ['artist2 song5', 'artist2 song6'],
        }),
    },
    'artist3': {
        'train': Dataset.from_dict({
            'text': ['artist3 song1', 'artist3 song3', 'artist3 song4'],
        }),
        'validation': Dataset.from_dict({
            'text': ['artist3 song5', 'artist3 song6'],
        }),
    },
}

TEST_DATA_TOKENIZED = {
    'artist1': {
        'train': Dataset.from_dict({
            'input_ids': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }),
        'validation': Dataset.from_dict({
            'input_ids': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }),
    },
    'artist2': {
        'train': Dataset.from_dict({
            'input_ids': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }),
        'validation': Dataset.from_dict({
            'input_ids': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }),
    },
    'artist3': {
        'train': Dataset.from_dict({
            'input_ids': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }),
        'validation': Dataset.from_dict({
            'input_ids': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }),
    },
}


# Write the test suite
class TestDataProcessing(unittest.TestCase):
    # Test the get_and_clean_data function
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        print("Temporarily creating a folder to store the test data.")
        os.makedirs(os.path.join(TEST_DATA_RAW_ROOT), exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
    #
    # def __del__(self):
    #     print("Cleaning up the test data")
    #     shutil.rmtree(TEST_DATA_ROOT)

    def test_get_and_clean_data(self):
        for artist in TEST_ARTISTS:
            # save the raw data
            TEST_DATA_RAW['artist1'].save_to_disk(os.path.join(TEST_DATA_RAW_ROOT, artist))
            # Run the function
            get_and_clean_data(artist, TEST_DATA_RAW_ROOT, TEST_DATA_CLEAN_ROOT, TEST_DATA_RAW[artist])

        # Check the output
        for artist in TEST_ARTISTS:
            ds_clean = Dataset.load_from_disk(os.path.join(TEST_DATA_CLEAN_ROOT, artist, 'train'))
            self.assertEqual(ds_clean['text'], TEST_DATA_CLEAN[artist]['train']['text'])

    # Test the split_datasets function
    def test_split_datasets(self):
        for artist in TEST_ARTISTS:
            # Run the function
            split_datasets(
                valid_size=0.4,
                load_dir=os.path.join(TEST_DATA_CLEAN_ROOT, artist),
                save_dir=os.path.join(TEST_DATA_SPLIT_ROOT, artist),
                non_random=True,
            )

        # Check the output
        for artist in TEST_ARTISTS:
            ds_split = load_from_disk(os.path.join(TEST_DATA_SPLIT_ROOT, artist))
            self.assertEqual(ds_split['train']['text'], TEST_DATA_SPLIT[artist]['train']['text'])
            self.assertEqual(ds_split['validation']['text'], TEST_DATA_SPLIT[artist]['validation']['text'])

    # Test the tokenize_datasets function
    def test_tokenize_datasets(self):
        tokenized_datasets = {}
        for artist in TEST_ARTISTS:
            ds_split = load_from_disk(os.path.join(TEST_DATA_SPLIT_ROOT, artist))
            # Run the function
            tokenizer_fn = partial(tokenize, block_size=4, stride=2)
            tokenized_datasets[artist] = ds_split.map(tokenizer_fn, batched=True, batch_size=1, remove_columns=['text'])
            tokenized_datasets[artist].save_to_disk(os.path.join(TEST_DATA_TOKENIZED_ROOT, artist))

        # Check the output
        for artist in TEST_ARTISTS:
            # check if the decoded tokenized datasets are the same as the original datasets
            ds_tokenized = load_from_disk(os.path.join(TEST_DATA_TOKENIZED_ROOT, artist))
            decoded_train = [self.tokenizer.decode(x) for x in ds_tokenized['train']['input_ids']]
            decoded_valid = [self.tokenizer.decode(x) for x in ds_tokenized['validation']['input_ids']]
            self.assertEqual(decoded_train, TEST_DATA_SPLIT[artist]['train']['text'])
            self.assertEqual(decoded_valid, TEST_DATA_SPLIT[artist]['validation']['text'])


if __name__ == '__main__':
    unittest.main()
