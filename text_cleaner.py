from nltk.tokenize import word_tokenize
from textblob import Word


class TextCleaner():
    def __init__(self, confidence: float = 0.85):
        self.confidence = confidence

    def correct_token(self, token: str) -> str:
        spellchecker = Word(token.lower())
        check_result = spellchecker.spellcheck()

        if check_result[0][1] > self.confidence:
            return check_result[0][0]

        return token

    def correct_sentence(self, sentence: str) -> str:
        tokens = word_tokenize(sentence)
        processed_tokens = []

        for token in tokens:
            processed_tokens.append(self.correct_token(token))

        return ' '.join(processed_tokens)
