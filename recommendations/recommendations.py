import json
from abc import ABC, abstractmethod
from typing import Callable
import pandas as pd
import numpy as np
from .text_cleaner import TextCleaner
from .duplicate_searcher import DuplicateSearcher


class AbcRecommendationsEngine(ABC):
    """Abstract interface for recsys"""
    @abstractmethod
    def get_recommendations(self, query: str, topn: int) -> list[str]:
        """Phrase continues pipeline"""

    @abstractmethod
    def show_brands(self, query: str, topn: int) -> list[str]:
        """Items searching pipeline"""

    @abstractmethod
    def get_recommendations_and_brands(self, query: str, topn: int) -> str:
        """Union of get_recommendations and show_brands methods"""


class RecommendationsEngine(AbcRecommendationsEngine):
    """Recsys implementation"""

    def __init__(self, vocab_filename: str,
                 vocab_row: str,
                 occurrence_row: str,
                 brand_row: str,
                 rating_row: str,
                 tokenizer: Callable,
                 similarity_threshold: float):
        self.vocab_row = vocab_row
        self.occurrence_row = occurrence_row
        self.brand_row = brand_row
        self.rating_row = rating_row

        self.data = pd.read_csv(vocab_filename)
        self.raw_data = self.data.copy()
        self.data = self.data.drop_duplicates(vocab_row)

        preprocessor = DuplicateSearcher(tokenizer)

        self.data = preprocessor.replace_duplicates_by_original(self.data, vocab_row,
                                                                occurrence_row, similarity_threshold)
        self.vocab = self.data[vocab_row].to_numpy(dtype='str')
        self.occurrences = self.data[occurrence_row].to_numpy()
        self.text_cleaner = TextCleaner()

    def __search_by_prefix(self, query: str) -> np.ndarray:
        """Find all sentences starts with given prefix"""

        matched_idx = np.where(np.char.startswith(self.vocab, query))[0]
        return matched_idx

    def __get_most_popular_queries_from_indices(self, tokens_idx: np.ndarray, topn: int) -> list[dict]:
        """Get sorted by popularity list from given tokens indices"""

        query_with_occurrence = {self.vocab[key]: self.occurrences[key] for key in tokens_idx}
        return sorted(query_with_occurrence.items(), key=lambda x: x[1], reverse=True)[:topn]

    def __get_most_popular_queries(self, query: str, topn: int = 5) -> list[dict]:
        """Get sorted by popularity list of phrases"""

        matched_idx = self.__search_by_prefix(query)
        most_popular = self.__get_most_popular_queries_from_indices(matched_idx, topn)
        return most_popular

    def __clean_recommendations(self, queries: list[dict]):
        """Spelling mistakes checker"""

        cleaned_most_popular = []
        for value, _ in queries:
            cleaned_most_popular.append(self.text_cleaner.correct_sentence(value))
        return cleaned_most_popular

    def get_recommendations(self, query: str, topn: int = 5) -> list[str]:
        """Phrase continues pipeline"""

        most_popular = self.__get_most_popular_queries(query, topn)
        cleaned_most_popular = self.__clean_recommendations(most_popular)
        return cleaned_most_popular

    def __get_most_popular_brands_from_indices(self, tokens_idx: np.ndarray, topn: int) -> list[str]:
        """Get sorted by popularity list of items from given tokens indices"""

        matched_items = self.raw_data[self.raw_data[self.vocab_row].isin(self.vocab[tokens_idx])]
        matched_items = matched_items.sort_values(by=[self.rating_row, self.occurrence_row],
                                                  ascending=False).reset_index()
        matched_items = matched_items[[self.brand_row, self.rating_row]][:topn]
        return matched_items[self.brand_row][:topn].to_list()

    def __get_most_popular_brands(self, query: str, topn: int) -> list[str]:
        """Get sorted by popularity list items"""

        matched_idx = self.__search_by_prefix(query)
        most_popular = self.__get_most_popular_brands_from_indices(matched_idx, topn)
        return most_popular

    def show_brands(self, query: str, topn: int) -> list[str]:
        """Items searching pipeline"""

        return self.__get_most_popular_brands(query, topn)

    def get_recommendations_and_brands(self, query: str, topn: int) -> str:
        """Union of get_recommendations and show_brands methods"""

        matched_idx = self.__search_by_prefix(query)
        most_popular_queries = self.__get_most_popular_queries_from_indices(matched_idx, topn)
        most_popular_queries = self.__clean_recommendations(most_popular_queries)
        most_popular_brands = self.__get_most_popular_brands_from_indices(matched_idx, topn)

        response = {'top_phrases': most_popular_queries, 'top_items': most_popular_brands}
        return json.dumps(response)
