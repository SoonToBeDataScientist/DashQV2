from __future__ import annotations

import functools


@functools.lru_cache(maxsize=4)
def get_scorer(backend: str = "auto", model_name: str = "ProsusAI/finbert"):
    """Return callable(list[str]) -> list[float] in [-1, 1]. Falls back to VADER."""
    if backend in ("auto", "finbert"):
        try:
            return _finbert(model_name)
        except Exception as e:
            if backend == "finbert":
                raise RuntimeError(f"FinBERT unavailable: {e}")
    return _vader()


def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    return lambda texts: [a.polarity_scores(t)["compound"] for t in texts]


def _finbert(model_name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    net = AutoModelForSequenceClassification.from_pretrained(model_name)
    net.eval()
    labels = [net.config.id2label[i].lower() for i in range(net.config.num_labels)]
    pos = [i for i, l in enumerate(labels) if "pos" in l]
    neg = [i for i, l in enumerate(labels) if "neg" in l]

    def score(texts, bs=16):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = tok(texts[i:i + bs], truncation=True, max_length=256,
                          padding=True, return_tensors="pt")
                p = torch.softmax(net(**enc).logits, dim=-1)
                out.extend((p[:, pos].sum(1) - p[:, neg].sum(1)).tolist())
        return [float(x) for x in out]
    return score
