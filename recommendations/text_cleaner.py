from autocorrect import Speller


class TextCleaner():
    """Text correction class"""
    def __init__(self):
        self.checker = Speller(fast=True)

    def correct_sentence(self, sentence: str) -> str:
        sentence = ' '.join(sentence.split())
        return self.checker(sentence)
