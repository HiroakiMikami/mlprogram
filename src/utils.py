import heapq
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re


class TopKElement:
    def __init__(self, k):
        self.k = k
        self._id = 0
        self.queue = []

    def push(self, score, get_data):
        #if len(self.queue) == self.k and score < self.queue[0][0]:
        #    return
        heapq.heappush(self.queue, [score, self._id, get_data()])
        self._id += 1
        while len(self.queue) > self.k:
            x = heapq.heappop(self.queue)


def bleu4(reference, candidate):
    sm = SmoothingFunction()

    def tokenize(code):
        code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'["\']', '`', code)
        tokens = [t for t in code.split(' ') if t]
        return tokens

    ref = tokenize(reference)
    cand = tokenize(candidate)
    return sentence_bleu([ref],
                         cand,
                         weights=[0.25] * min(4, len(ref)),
                         smoothing_function=sm.method3)
