from gensim.models.fasttext import FastText
import pandas as pd
import numpy as np
import faiss
from typing import Callable


class Sent2Vec():
    def __init__(self, tokenizer: Callable):
        self.model = FastText(vector_size=100)
        self.tokenizer = tokenizer

    def train(self, sentences: pd.Series):
        tokenized_sentences = sentences.apply(self.tokenizer)
        self.model.build_vocab(corpus_iterable=tokenized_sentences)
        self.model.train(corpus_iterable=tokenized_sentences, total_examples=self.model.corpus_count,
                         epochs=self.model.epochs)

    def __get_word_embedding(self, word: str) -> np.ndarray:
        return self.model.wv[word]

    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        tokens = self.tokenizer(sentence)
        sentence_embedding = np.zeros(self.model.vector_size)

        for token in tokens:
            sentence_embedding += self.__get_word_embedding(token)

        return sentence_embedding / len(tokens) / np.linalg.norm(sentence_embedding)


class DuplicateSearcher():
    def __init__(self, tokenizer: Callable):
        self.vectorizer = Sent2Vec(tokenizer)

    def __vectorize_corpus(self, sentences: pd.Series) -> (faiss.IndexFlatL2, np.ndarray):
        vectorized_corpus = np.zeros((len(sentences), self.vectorizer.model.vector_size))

        for i in range(len(sentences)):
            vectorized_corpus[i] = self.vectorizer.get_sentence_embedding(sentences.iloc[i])

        index = faiss.IndexFlatIP(self.vectorizer.model.vector_size)
        index.add(vectorized_corpus)

        return index, vectorized_corpus

    def replace_duplicates_by_original(self, corpus: pd.DataFrame, vocab_row: str,
                                       occurrences_row: str, similarity_threshold: float = 0.85) -> pd.DataFrame:
        corpus = corpus.drop_duplicates(vocab_row)
        corpus = corpus.sort_values(by=[occurrences_row, vocab_row], ascending=False).reset_index()

        self.vectorizer.train(corpus[vocab_row])

        index, vectorized_corpus = self.__vectorize_corpus(corpus[vocab_row])

        i = 0
        while i < len(corpus):
            closest = index.search(vectorized_corpus[i][np.newaxis, :], 10)
            similarity = closest[0][0][1:]
            idx = closest[1][0][1:]
            for sim, j in zip(similarity, idx):
                if corpus.loc[j, occurrences_row] == 0:
                    continue
                if sim > similarity_threshold:
                    corpus.loc[i, occurrences_row] = corpus.loc[i, occurrences_row] + corpus.loc[j, occurrences_row]
                    corpus.loc[j, occurrences_row] = 0

                else:
                    break

            i += 1
        corpus.drop(corpus[corpus[occurrences_row] == 0].index, inplace=True)
        return corpus
