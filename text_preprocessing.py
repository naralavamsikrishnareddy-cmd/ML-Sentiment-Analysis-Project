import re

NEGATION_WORDS = {
    "no","not","nor","never","neither","nothing","nowhere","nobody",
    "hardly","barely","scarcely","without","isn't","wasn't","aren't",
    "weren't","don't","doesn't","didn't","can't","couldn't","won't",
    "wouldn't","shouldn't"
}

STOPWORDS = {
    "the","a","an","and","or","but","is","are","was","were","be","been",
    "to","of","for","in","on","at","with","this","that","it","its","as",
    "my","i","we","they","he","she","you","very","really","just","from",
    "by","our","their","have","has","had","am","so","me"
}

def normalize_text(text):
    text = "" if text is None else str(text).lower()
    text = text.replace("’", "'")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalize_text(text))

def preprocess_review(text):
    tokens = tokenize(text)
    kept = []
    for token in tokens:
        if len(token) > 1 and (token in NEGATION_WORDS or token not in STOPWORDS):
            kept.append(token)
    # Keep bigrams because they help phrases such as "not good".
    bigrams = [kept[i] + "_" + kept[i+1] for i in range(len(kept)-1)]
    return kept + bigrams
