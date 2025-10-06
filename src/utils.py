# src/utils.py
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

def eval_classification(model, X, y, feature_names=None):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else preds
    print("Classification report:")
    print(classification_report(y, preds))
    try:
        print("ROC AUC:", roc_auc_score(y, probs))
    except Exception:
        pass
    if feature_names:
        try:
            fmap = model.get_booster().get_score(importance_type="gain")
            items = sorted(fmap.items(), key=lambda x: x[1], reverse=True)
            print("Top features (gain):")
            for k, v in items[:20]:
                print(k, v)
        except Exception:
            pass
