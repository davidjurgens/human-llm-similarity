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
        return [self._get_subjectivity(text) for text in texts]
    

if __name__ == "__main__":
    analyzer = SubjectivityAnalyzer()
    texts = ["I love programming in Python.", "I hate programming in Python."]
    scores = analyzer.get_subjectivity_scores(texts)
    print(scores)