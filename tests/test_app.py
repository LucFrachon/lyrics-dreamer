# Write a test suite for the app.py module

import unittest

from app import load_artists


TEST_ARTISTS = ['name1', 'compound-name2', 'compound_name3', 'complex_compound-name_4']
TARGET_ARTISTS = ['Name1', 'Compound Name2', 'Compound Name3', 'Complex Compound Name 4']


class TestApp(unittest.TestCase):
    def test_load_artists(self):
        artists = load_artists(TEST_ARTISTS)  # dict of {'artist_id': 'artist_name'}
        print(artists)
        clean_artists = [artists[name] for name in TEST_ARTISTS]
        self.assertEqual(clean_artists, TARGET_ARTISTS)

if __name__ == '__main__':
    unittest.main()
