from typing import List
from textblob import TextBlob


class SubjectivityAnalyzer:
    def __init__(self):
        self.blob = TextBlob

    def _get_subjectivity(self, text: str) -> float:
        """
        Get the subjectivity score of a text. 0 being objective and 1 subjective.
        """
        return self.blob(text).sentiment.subjectivity
    
    def get_subjectivity_scores(self, texts: List[str]) -> List[float]:
        """
        Get the subjectivity scores of a list of texts.
        """
        scores = [self._get_subjectivity(text) for text in texts]
        return [max(score, 1e-10) for score in scores]  # Ensure no zero values


if __name__ == "__main__":
    analyzer = SubjectivityAnalyzer()
    texts = ["I love programming in C++.", "I hate programming in C++."]
    scores = analyzer.get_subjectivity_scores(texts)
    print(scores)