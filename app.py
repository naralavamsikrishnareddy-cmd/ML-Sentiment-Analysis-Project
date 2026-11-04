from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
from collections import Counter
import csv, io, json, re
from ml_engine import load_data, load_bundle, train_ensemble, MODEL_PATH, METRICS_PATH, predict_texts
from text_preprocessing import preprocess_review

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "reviews.csv"

DOMAIN_KEYWORDS = {
    "Product": {"phone","mobile","smartphone","laptop","camera","battery","screen","display","device","charger","keyboard","mouse","product","tablet","headphones","earbuds","performance","delivery","software","app"},
    "Movie": {"movie","film","story","plot","acting","actor","actress","direction","director","music","soundtrack","cinematography","screenplay","climax","characters","visuals","effects","ending","scene","dialogue","cast","episode"},
    "Restaurant": {"restaurant","food","biryani","taste","staff","service","dish","meal","dessert","ambience","hygiene","portion","menu","cafe","dining","waiter","cuisine","spicy","order"}
}
ASPECTS = {
    "Product": {"Camera":["camera","photo","photos","picture","pictures"],"Battery":["battery","charging","charge","drain","drains"],"Display":["screen","display","touch"],"Performance":["performance","fast","slow","freeze","freezes"],"Build Quality":["build","quality","design"],"Delivery":["delivery","package","arrived"],"Price":["price","cost","value"]},
    "Movie": {"Story":["story","plot","screenplay"],"Acting":["acting","actor","actress","performance","cast"],"Direction":["direction","director"],"Music":["music","soundtrack","song"],"Visuals":["visual","visuals","cinematography","effects"],"Characters":["character","characters"],"Pacing":["pacing"],"Ending":["ending","climax"]},
    "Restaurant": {"Food & Taste":["food","taste","biryani","dish","meal","dessert","cuisine"],"Service":["service","staff","waiter"],"Ambience":["ambience","atmosphere","noise","noisy","comfortable"],"Hygiene":["hygiene","clean","cleanliness","dirty"],"Portion":["portion","size"],"Price":["price","cost","value"]}
}

def domain_detect(text):
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    scores = {d: len(tokens & words) for d, words in DOMAIN_KEYWORDS.items()}
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ordered or ordered[0][1] < 1:
        return "Unknown", scores
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1] and ordered[0][1] > 0:
        return "Ambiguous", scores
    return ordered[0][0], scores

def ensure_model():
    if not MODEL_PATH.exists():
        train_ensemble(load_data())

def analyze(text, chosen="Auto-detect"):
    text = str(text).strip()
    if len(text.split()) < 4:
        raise ValueError("Please enter at least 4 words.")
    detected, scores = domain_detect(text)
    domain = detected if chosen == "Auto-detect" else chosen
    if chosen == "Auto-detect" and domain in {"Unknown", "Ambiguous"}:
        raise ValueError("Please add Product, Movie, or Restaurant context so the domain can be identified.")
    ensure_model()
    bundle = load_bundle()
    result = predict_texts(bundle, [text])
    p = result["probabilities"][0]
    idx = max(range(3), key=lambda i: p[i])
    label = result["classes"][idx]
    confidence = p[idx]

    aspects = []
    # Score the clause containing each aspect, rather than a broad window that
    # mixes positive camera feedback with negative battery feedback.
    clauses = [c.strip(" ,.!?;:") for c in re.split(
        r"\b(?:but|however|although|though|yet|whereas|while)\b", text,
        flags=re.IGNORECASE
    ) if c.strip(" ,.!?;:")]
    positive_words = {
        "excellent", "amazing", "great", "good", "fantastic", "wonderful",
        "love", "loved", "perfect", "clear", "sharp", "fast", "smooth",
        "beautiful", "reliable", "recommend", "satisfied", "durable", "easy"
    }
    negative_words = {
        "bad", "poor", "terrible", "awful", "worst", "hate", "hated",
        "slow", "drain", "drains", "drained", "quickly", "broken", "blurry",
        "lag", "laggy", "issue", "problem", "disappointing", "disappointed",
        "faulty", "weak", "short", "overheat", "overheats", "expensive"
    }
    negators = {"not", "never", "no", "hardly", "isn't", "wasn't", "doesn't", "don't"}
    def aspect_polarity(clause):
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", clause.lower())
        score = 0
        for i, word in enumerate(words):
            val = 1 if word in positive_words else (-1 if word in negative_words else 0)
            if val and any(w in negators for w in words[max(0, i-3):i]):
                val *= -1
            score += val
        if score > 0:
            return "positive", min(0.96, 0.72 + 0.06 * min(score, 4))
        if score < 0:
            return "negative", min(0.96, 0.72 + 0.06 * min(abs(score), 4))
        return None, 0.0

    if domain in ASPECTS:
        for aspect, keys in ASPECTS[domain].items():
            matched_clause = None
            for clause in clauses:
                clause_tokens = set(re.findall(r"[a-z0-9]+", clause.lower()))
                if clause_tokens.intersection(keys):
                    matched_clause = clause
                    break
            if matched_clause:
                context = matched_clause
                rule_label, rule_conf = aspect_polarity(context)
                if rule_label is not None:
                    aspect_label, aspect_conf = rule_label, rule_conf
                else:
                    ar = predict_texts(bundle, [context])
                    ap = ar["probabilities"][0]
                    ai = max(range(3), key=lambda i: ap[i])
                    aspect_label, aspect_conf = ar["classes"][ai], ap[ai]
                aspects.append({"aspect":aspect,"sentiment":aspect_label,"confidence":aspect_conf,"context":context})
    labs = {a["sentiment"] for a in aspects}
    if "positive" in labs and "negative" in labs:
        label = "mixed"
    elif confidence < 0.50:
        label = "uncertain"

    models = []
    for name, obj in result["models"].items():
        mi = max(range(3), key=lambda i: obj["probabilities"][0][i])
        models.append({"name":name,"prediction":obj["labels"][0],"confidence":obj["probabilities"][0][mi]})
    return {"domain":domain,"domain_scores":scores,"sentiment":label,"confidence":confidence,"models":models,"probabilities":dict(zip(result["classes"],p)),"preprocessed":" ".join(preprocess_review(text)),"aspects":aspects}

@app.get("/")
def index():
    rows = load_data()
    counts = Counter(r["sentiment"].lower() for r in rows)
    domains = Counter(r.get("category", "Unknown") for r in rows)
    ensure_model()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return render_template("index.html", total=len(rows), counts=counts, domains=domains, rows=rows[:12], metrics=metrics)

@app.post("/analyze")
def analyze_route():
    try:
        data = request.get_json(silent=True) or request.form
        return jsonify({"ok":True,"result":analyze(data.get("text", ""), data.get("domain", "Auto-detect"))})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 400

@app.get("/metrics")
def metrics():
    ensure_model()
    return jsonify(json.loads(METRICS_PATH.read_text(encoding="utf-8")))

@app.post("/retrain")
def retrain():
    try:
        _, metrics = train_ensemble(load_data())
        return jsonify({"ok":True,"metrics":metrics})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 400

@app.post("/upload-training")
def upload_training():
    file = request.files.get("file")
    if not file:
        return jsonify({"ok":False,"error":"Choose a CSV file."}), 400
    try:
        raw = file.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(raw))
        fields = reader.fieldnames or []
        required = {"review", "sentiment"}
        if not required.issubset(set(fields)):
            return jsonify({"ok":False,"error":"Training CSV must contain review and sentiment columns."}), 400
        rows = list(reader)
        if len(rows) < 30:
            return jsonify({"ok":False,"error":"Training CSV should contain at least 30 labeled reviews."}), 400
        DATA_PATH.write_text(raw, encoding="utf-8")
        _, metrics = train_ensemble(load_data())
        return jsonify({"ok":True,"message":f"Uploaded {len(rows)} reviews and retrained the model.","metrics":metrics})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 400

@app.post("/batch")
def batch():
    file = request.files.get("file")
    if not file:
        return jsonify({"ok":False,"error":"Choose a CSV file."}), 400
    try:
        ensure_model(); bundle = load_bundle()
        raw = file.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(raw))
        if "review" not in (reader.fieldnames or []):
            return jsonify({"ok":False,"error":"CSV must contain a review column."}), 400
        rows = []
        for r in reader:
            text = str(r.get("review", ""))
            if len(text.split()) >= 4:
                result = predict_texts(bundle, [text]); i = max(range(3), key=lambda j: result["probabilities"][0][j])
                r["detected_domain"] = domain_detect(text)[0]
                r["predicted_sentiment"] = result["classes"][i]
                r["confidence"] = f"{result['probabilities'][0][i]:.4f}"
            else:
                r["detected_domain"] = "Unknown"; r["predicted_sentiment"] = "unsupported"; r["confidence"] = "0"
            rows.append(r)
        output = io.StringIO(); fields = list(rows[0].keys()) if rows else ["review","predicted_sentiment","confidence"]
        writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        return send_file(io.BytesIO(output.getvalue().encode("utf-8")), mimetype="text/csv", as_attachment=True, download_name="review_analysis_results.csv")
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 400

if __name__ == "__main__":
    ensure_model()
    app.run(host="127.0.0.1", port=5000, debug=True)
