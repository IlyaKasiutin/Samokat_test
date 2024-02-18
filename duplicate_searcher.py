import gensim
import gensim.downloader
import pandas as pd


class DuplicateSearcher():
    def __init__(self, w2v_model: str):
        self.model = gensim.downloader.load(w2v_model)

    def replace_duplicates_by_original(self, corpus: pd.DataFrame, vocab_row: str,
                                       occurrences_row: str, similarity_threshold: float = 0.85) -> pd.DataFrame:
        corpus = corpus.drop_duplicates(vocab_row)
        corpus = corpus.sort_values(by=[occurrences_row, vocab_row], ascending=False)

        i = 0
        while i < len(corpus):
            closest = self.model.most_similar(corpus[i])
            for value, similarity in closest:
                if similarity > similarity_threshold:
                    corpus.iloc[i][occurrences_row] += corpus[corpus[vocab_row] == value][occurrences_row]
                    corpus = corpus.drop(corpus[corpus[vocab_row] == value].index)
                else:
                    break
            i += 1
        return corpus
