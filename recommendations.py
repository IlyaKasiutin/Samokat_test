import pandas as pd
import numpy as np
from typing import Callable
from text_cleaner import TextCleaner
from duplicate_searcher import DuplicateSearcher


class RecommendationsEngine():
    def __init__(self, vocab_filename: str, vocab_row: str, occurrence_row: str,
                 tokenizer: Callable, similarity_threshold: float):
        data = pd.read_csv(vocab_filename)
        preprocessor = DuplicateSearcher(tokenizer)
        data = data.drop_duplicates(vocab_row)
        data = preprocessor.replace_duplicates_by_original(data, vocab_row, occurrence_row, similarity_threshold)
        self.vocab = data[vocab_row].to_numpy(dtype='str')
        self.occurrences = data[occurrence_row].to_numpy()
        self.text_cleaner = TextCleaner()

    def __search_by_prefix(self, query: str) -> np.ndarray:
        matched_idx = np.where(np.char.startswith(self.vocab, query))[0]
        return matched_idx

    def __get_most_popular(self, tokens_idx: np.ndarray, topn: int) -> list:
        query_with_occurrence = {self.vocab[key]: self.occurrences[key] for key in tokens_idx}
        return sorted(query_with_occurrence.items(), key=lambda x: x[1], reverse=True)[:topn]

    def get_recommendations(self, query: str, topn: int = 5) -> list[str]:
        matched_idx = self.__search_by_prefix(query)
        most_popular = self.__get_most_popular(matched_idx, topn)

        cleaned_most_popular = []
        for value, _ in most_popular:
            cleaned_most_popular.append(self.text_cleaner.correct_sentence(value))
        return cleaned_most_popular

    def show_brands(self, query: str, topn: int) -> np.ndarray:
        pass
