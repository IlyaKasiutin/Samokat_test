from abc import ABC, abstractmethod


class AbcRecommendationsEngine(ABC):
    @abstractmethod
    def get_recommendations(self, query: str, topn: int) -> list[str]:
        pass

    @abstractmethod
    def show_brands(self, query: str, topn: int) -> list[str]:
        pass
