import stanfordnlp


class PosTagger:
    def __init__(self):
        nlp = stanfordnlp.Pipeline(lang='id')
        self.nlp = nlp
    def tag(self, sentence):
        doc = self.nlp(sentence)
        pos = [(word.text, word.upos) for sent in doc.sentences for word in sent.words]
        return pos