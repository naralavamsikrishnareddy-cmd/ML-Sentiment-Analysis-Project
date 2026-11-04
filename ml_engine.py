"""
Pure-Python sentiment engine.

This version intentionally does not import NumPy, SciPy, pandas, scikit-learn,
or joblib. That makes it suitable for Windows systems where compiled Python
extensions are blocked by an Application Control policy.

Models:
1. Multinomial Naive Bayes
2. Logistic Regression (one-vs-rest SGD)
3. Linear SVM (one-vs-rest hinge-loss SGD)
4. Lightweight Random Forest (bagged random decision stumps)
5. Soft-voting ensemble
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, math, pickle, random
from text_preprocessing import preprocess_review

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model_bundle.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
CLASSES = ["negative", "neutral", "positive"]

class TFIDF:
    def __init__(self, max_features=2500):
        self.max_features = max_features
        self.vocab = {}
        self.idf = []

    def fit(self, docs):
        df = Counter()
        counts_all = []
        for doc in docs:
            c = Counter(preprocess_review(doc))
            counts_all.append(c)
            for term in c:
                df[term] += 1
        ranked = sorted(df.items(), key=lambda x: (-x[1], x[0]))[:self.max_features]
        self.vocab = {term:i for i,(term,_) in enumerate(ranked)}
        n = len(docs)
        self.idf = [0.0] * len(self.vocab)
        for term,i in self.vocab.items():
            self.idf[i] = math.log((1+n)/(1+df[term])) + 1.0
        return self

    def transform(self, docs):
        result=[]
        for doc in docs:
            c=Counter(preprocess_review(doc))
            vec={}
            norm=0.0
            for term,count in c.items():
                i=self.vocab.get(term)
                if i is not None:
                    value=(1.0+math.log(count))*self.idf[i]
                    vec[i]=value
                    norm += value*value
            norm=math.sqrt(norm) or 1.0
            result.append({i:v/norm for i,v in vec.items()})
        return result

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

class NaiveBayes:
    def fit(self, X, y):
        self.class_doc_counts=Counter(y)
        self.term_counts={c:Counter() for c in CLASSES}
        self.total_terms=Counter()
        for vec,label in zip(X,y):
            for i,v in vec.items():
                self.term_counts[label][i]+=v
                self.total_terms[label]+=v
        self.vocab_size=max((max((max(x.keys()) for x in X if x), default=-1)+1), 1)
        self.class_log_prior={c:math.log((self.class_doc_counts[c]+1)/(len(y)+3)) for c in CLASSES}
        return self
    def predict_proba(self,X):
        out=[]
        for vec in X:
            scores=[]
            for c in CLASSES:
                denom=self.total_terms[c]+self.vocab_size
                s=self.class_log_prior[c]
                for i,v in vec.items():
                    s += v*math.log((self.term_counts[c][i]+1)/denom)
                scores.append(s)
            out.append(_softmax(scores))
        return out

class LogisticRegressionSGD:
    def __init__(self, epochs=12, lr=0.08, l2=0.0005):
        self.epochs=epochs; self.lr=lr; self.l2=l2
    def fit(self,X,y):
        self.weights={c:defaultdict(float) for c in CLASSES}
        self.bias={c:0.0 for c in CLASSES}
        for epoch in range(self.epochs):
            order=list(range(len(y))); random.Random(42+epoch).shuffle(order)
            rate=self.lr/(1+0.15*epoch)
            for k in order:
                scores=[self.bias[c]+sum(self.weights[c][i]*v for i,v in X[k].items()) for c in CLASSES]
                probs=_softmax(scores)
                for j,c in enumerate(CLASSES):
                    error=(1.0 if y[k]==c else 0.0)-probs[j]
                    for i,v in X[k].items():
                        self.weights[c][i] += rate*(error*v-self.l2*self.weights[c][i])
                    self.bias[c] += rate*error
        return self
    def predict_proba(self,X):
        return [_softmax([self.bias[c]+sum(self.weights[c][i]*v for i,v in vec.items()) for c in CLASSES]) for vec in X]

class LinearSVM:
    def __init__(self, epochs=10, lr=0.04, l2=0.0005):
        self.epochs=epochs; self.lr=lr; self.l2=l2
    def fit(self,X,y):
        self.weights={c:defaultdict(float) for c in CLASSES}
        self.bias={c:0.0 for c in CLASSES}
        for epoch in range(self.epochs):
            order=list(range(len(y))); random.Random(100+epoch).shuffle(order)
            rate=self.lr/(1+0.12*epoch)
            for k in order:
                for c in CLASSES:
                    target=1.0 if y[k]==c else -1.0
                    margin=target*(self.bias[c]+sum(self.weights[c][i]*v for i,v in X[k].items()))
                    if margin < 1.0:
                        for i,v in X[k].items():
                            self.weights[c][i] += rate*(target*v-self.l2*self.weights[c][i])
                        self.bias[c] += rate*target
                    else:
                        for i in X[k]:
                            self.weights[c][i] *= (1-rate*self.l2)
        return self
    def predict_proba(self,X):
        # Convert class scores into a probability-like distribution for voting.
        return [_softmax([self.bias[c]+sum(self.weights[c][i]*v for i,v in vec.items()) for c in CLASSES]) for vec in X]

class RandomStumpForest:
    """A small pure-Python bagged forest of random one-feature decision stumps."""
    def __init__(self, trees=80, sample_ratio=0.75):
        self.trees=trees; self.sample_ratio=sample_ratio; self.stumps=[]
    def fit(self,X,y):
        rng=random.Random(42)
        n=len(X)
        features=list({i for vec in X for i in vec})
        self.stumps=[]
        for _ in range(self.trees):
            if not features: break
            f=rng.choice(features)
            idx=[rng.randrange(n) for _ in range(max(20,int(n*self.sample_ratio)))]
            present=Counter(); absent=Counter()
            for k in idx:
                if f in X[k]: present[y[k]]+=1
                else: absent[y[k]]+=1
            self.stumps.append((f,_distribution(present),_distribution(absent)))
        return self
    def predict_proba(self,X):
        if not self.stumps:
            return [[1/3,1/3,1/3] for _ in X]
        out=[]
        for vec in X:
            p=[0.0,0.0,0.0]
            for f,yes,no in self.stumps:
                q=yes if f in vec else no
                for i,v in enumerate(q): p[i]+=v
            total=sum(p) or 1.0
            out.append([v/total for v in p])
        return out

def _distribution(counter):
    total=sum(counter.values())+3
    return [(counter[c]+1)/total for c in CLASSES]

def _softmax(scores):
    m=max(scores)
    ex=[math.exp(max(-60,min(60,s-m))) for s in scores]
    z=sum(ex) or 1.0
    return [v/z for v in ex]

def _predict_labels(probs):
    return [CLASSES[max(range(3),key=lambda i:p[i])] for p in probs]

def _accuracy(y,pred):
    return sum(a==b for a,b in zip(y,pred))/max(1,len(y))

def _metrics(y,pred):
    cm=[[0]*3 for _ in range(3)]
    for a,b in zip(y,pred):
        cm[CLASSES.index(a)][CLASSES.index(b)]+=1
    precision=[]; recall=[]; f1=[]
    for i in range(3):
        tp=cm[i][i]; fp=sum(cm[r][i] for r in range(3))-tp
        fn=sum(cm[i][c] for c in range(3))-tp
        pr=tp/(tp+fp) if tp+fp else 0
        rc=tp/(tp+fn) if tp+fn else 0
        precision.append(pr); recall.append(rc)
        f1.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return {
        "accuracy":_accuracy(y,pred),
        "precision_macro":sum(precision)/3,
        "recall_macro":sum(recall)/3,
        "f1_macro":sum(f1)/3,
        "classes":CLASSES,
        "confusion_matrix":cm
    }

def load_data(path=None):
    path=Path(path or ROOT/"data/reviews.csv")
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def train_ensemble(rows, test_size=0.20):
    work=[]
    for r in rows:
        review=str(r.get("review","")).strip()
        sentiment=str(r.get("sentiment","")).strip().lower()
        if len(review.split())>=3 and sentiment in CLASSES:
            work.append(r)
    if len(work)<30 or len(set(r["sentiment"].lower() for r in work))<3:
        raise ValueError("Need at least 30 labeled reviews containing all three classes.")
    rng=random.Random(42)
    idx=list(range(len(work))); rng.shuffle(idx)
    cut=max(3,int(len(work)*(1-test_size)))
    train_idx, test_idx=idx[:cut],idx[cut:]
    train_rows=[work[i] for i in train_idx]; test_rows=[work[i] for i in test_idx]
    vectorizer=TFIDF(max_features=2500)
    X_train=vectorizer.fit_transform([r["review"] for r in train_rows])
    X_test=vectorizer.transform([r["review"] for r in test_rows])
    y_train=[r["sentiment"].lower() for r in train_rows]
    y_test=[r["sentiment"].lower() for r in test_rows]

    models={
        "Naive Bayes":NaiveBayes(),
        "Logistic Regression":LogisticRegressionSGD(),
        "SVM":LinearSVM(),
        "Random Forest":RandomStumpForest()
    }
    metrics={}
    test_probs=[]
    for name,model in models.items():
        model.fit(X_train,y_train)
        probs=model.predict_proba(X_test)
        test_probs.append(probs)
        metrics[name]=_metrics(y_test,_predict_labels(probs))
    ens=[]
    for k in range(len(X_test)):
        ens.append([sum(p[k][j] for p in test_probs)/len(test_probs) for j in range(3)])
    metrics["Soft Voting Ensemble"]=_metrics(y_test,_predict_labels(ens))

    # Refit all models on the full dataset for deployment.
    vectorizer=TFIDF(max_features=2500)
    X_all=vectorizer.fit_transform([r["review"] for r in work])
    y_all=[r["sentiment"].lower() for r in work]
    final_models={}
    for name,cls in [
        ("Naive Bayes",NaiveBayes),("Logistic Regression",LogisticRegressionSGD),
        ("SVM",LinearSVM),("Random Forest",RandomStumpForest)
    ]:
        model=cls(); model.fit(X_all,y_all); final_models[name]=model

    bundle={"vectorizer":vectorizer,"models":final_models,"classes":CLASSES,"version":"4.0-pure-python"}
    MODEL_DIR.mkdir(exist_ok=True)
    with MODEL_PATH.open("wb") as f: pickle.dump(bundle,f)
    payload={
        "version":bundle["version"],"dataset_rows":len(work),
        "train_rows":len(train_rows),"test_rows":len(test_rows),
        "class_counts":dict(Counter(y_all)),
        "category_counts":dict(Counter(r.get("category","Unknown") for r in work)),
        "metrics":metrics
    }
    METRICS_PATH.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return bundle,payload

def load_bundle():
    with MODEL_PATH.open("rb") as f: return pickle.load(f)

def predict_texts(bundle,texts):
    X=bundle["vectorizer"].transform(texts)
    outputs={}; probs=[]
    for name in ["Naive Bayes","Logistic Regression","SVM","Random Forest"]:
        p=bundle["models"][name].predict_proba(X)
        probs.append(p)
        outputs[name]={"labels":_predict_labels(p),"probabilities":p}
    ensemble=[]
    for k in range(len(X)):
        ensemble.append([sum(p[k][j] for p in probs)/len(probs) for j in range(3)])
    return {"labels":_predict_labels(ensemble),"probabilities":ensemble,"classes":CLASSES,"models":outputs}
