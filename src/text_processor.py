import multiprocessing as mp
import re
import string

import en_core_web_sm
import gensim.downloader as api
import numpy as np
import pandas as pd
import unidecode
from normalise import normalise
from pycontractions import Contractions
from sklearn.base import TransformerMixin, BaseEstimator


def setup():
    import nltk
    import subprocess
    nltk.download('brown')
    nltk.download('names')
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 variety="BrE",
                 user_abbrevs={},
                 n_jobs=1):
        """
        Text processor
        -removes extra whitespace within text
        -converts unicode to ascii
        -converts to lowercase
        -remove leading or trailing whitespace
        -expands contractions
        -tokenizes sentences and words
        -removes punctuation
        -lemmatizes words
        -removes stopwords

        :param variety: format of date (AmE - american type, BrE - british format)
        :param user_abbrevs: dict of user abbreviations mappings (from normalise package)
        :param n_jobs: parallel jobs to run

        adapted from Maksim Balatsko and Christina Levengood
        """

        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs
        self.cont = Contractions(kv_model=api.load("glove-twitter-25"))
        self.punctuation_table = str.maketrans('', '', string.punctuation)
        self.nlp = en_core_web_sm.load()

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        pool.close()
        pool.join()

        return data

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        text = re.sub(' +', ' ', text)
        text = unidecode.unidecode(text)
        text = text.lower()
        text = self._replace_contraction(text)
        text = self._clean_numbers(text)
        text = self._normalize(text)
        text = self.nlp(str(text))
        text = self._remove_punct(text)
        text = self._remove_stop_words(text)
        return self._clean(text)

    @staticmethod
    def _clean_numbers(x):
        if bool(re.search(r'\d', x)):
            x = re.sub('[0-9]{5,}', '#####', x)
            x = re.sub('[0-9]{4}', '####', x)
            x = re.sub('[0-9]{3}', '###', x)
            x = re.sub('[0-9]{2}', '##', x)
        return x

    def _normalize(self, text):
        try:
            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))
        except:
            return text

    def _replace_contraction(self, text):
        expanded_contractions = list(self.cont.expand_texts([text], precise=True))
        if expanded_contractions:
            text = expanded_contractions[0]
        return text

    @staticmethod
    def _remove_punct(doc):
        return [t for t in doc if t.text not in string.punctuation]

    @staticmethod
    def _remove_stop_words(doc):
        return [t for t in doc if not t.is_stop]

    @staticmethod
    def _clean(doc):
        return ' '.join([i.text for i in doc if i]).strip().lower()

    @staticmethod
    def _lemmatize(doc):
        return ' '.join([t.lemma_ for t in doc])


if __name__ == '__main__':
    x = "It's a test nvermind shouldn't sentence with different 200 ias chass to fix"
    x = TextPreprocessor(n_jobs=1).transform(pd.DataFrame.from_dict({"review": [x]}).review).values[0]
    print(x)
