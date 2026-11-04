from ml_engine import load_data, train_ensemble, MODEL_PATH
if __name__ == "__main__":
    print("Training pure-Python ensemble...")
    bundle, metrics = train_ensemble(load_data())
    print("\nTraining complete.")
    print(f"Dataset: {metrics['dataset_rows']} reviews")
    for name, m in metrics["metrics"].items():
        print(f"{name:22s} Accuracy={m['accuracy']:.2%}  F1={m['f1_macro']:.2%}")
    print(f"\nModel saved to: {MODEL_PATH}")
