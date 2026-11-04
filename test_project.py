from ml_engine import load_data, train_ensemble, predict_texts

if __name__ == "__main__":
    rows = load_data()
    bundle, metrics = train_ensemble(rows)
    result = predict_texts(bundle, [
        "The camera is excellent but the battery drains quickly.",
        "The food was delicious and the staff were friendly.",
        "The movie story was boring and the ending was disappointing."
    ])
    for text, label, probs in zip(
        ["Product example","Restaurant example","Movie example"],
        result["labels"], result["probabilities"]
    ):
        print(f"{text}: {label} {probs}")
    print("Project test completed.")
