from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("bot_detector.pkl", "rb") as f:
    bundle = pickle.load(f)

rf = bundle["rf"]
gb = bundle["gb"]
lr = bundle["lr"]
scaler = bundle["scaler"]
features = bundle["features"]
best_single = bundle["best_single"]

UI_FIELDS = [
    {
        "name": "account_age_days",
        "label": "Account age",
        "help": "How old is the account in days?",
        "type": "number",
        "step": "1",
        "min": "0",
        "max": "5000",
        "value": "180",
        "group": "Basic Profile",
    },
    {
        "name": "ff_ratio",
        "label": "Followers compared to following",
        "help": "Low means the account follows many more people than follow it back.",
        "type": "range",
        "step": "0.01",
        "min": "0",
        "max": "5",
        "value": "0.50",
        "group": "Basic Profile",
    },
    {
        "name": "avg_daily_posts",
        "label": "Posts per day",
        "help": "Average number of posts per day.",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "max": "500",
        "value": "1.0",
        "group": "Basic Profile",
    },
    {
        "name": "avg_hashtags_per_post",
        "label": "Hashtags per post",
        "help": "Average hashtags used in each post.",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "max": "50",
        "value": "2.0",
        "group": "Basic Profile",
    },
    {
        "name": "spam_keyword_score",
        "label": "Spammy words in bio/posts",
        "help": "Higher means more giveaway, promo, crypto, or suspicious language.",
        "type": "range",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "value": "0.10",
        "group": "Content Behavior",
    },
    {
        "name": "duplicate_post_ratio",
        "label": "Repeated posts",
        "help": "Higher means the account reuses the same or very similar content.",
        "type": "range",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "value": "0.10",
        "group": "Content Behavior",
    },
    {
        "name": "content_diversity",
        "label": "Variety of content",
        "help": "Higher means content is more varied.",
        "type": "range",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "value": "0.60",
        "group": "Content Behavior",
    },
    {
        "name": "sentiment_variance",
        "label": "Variation in tone",
        "help": "Higher means the tone of posts changes more.",
        "type": "range",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "value": "0.30",
        "group": "Content Behavior",
    },
    {
        "name": "burst_score",
        "label": "Posting pattern",
        "help": "Higher means many posts happen in short bursts instead of naturally over time.",
        "type": "range",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "value": "0.20",
        "group": "Activity Pattern",
    },
    {
        "name": "posting_hour_std",
        "label": "Posting time regularity",
        "help": "Lower means posts happen around the same times; higher means timing is more spread out.",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "max": "12",
        "value": "3.5",
        "group": "Activity Pattern",
    },
    {
        "name": "mutual_follow_rate",
        "label": "Mutual connections",
        "help": "Higher means more people follow each other back.",
        "type": "range",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "value": "0.30",
        "group": "Activity Pattern",
    },
    {
        "name": "name_entropy",
        "label": "Username randomness",
        "help": "Higher means the username looks more random or auto-generated.",
        "type": "range",
        "step": "0.01",
        "min": "0",
        "max": "5",
        "value": "2.0",
        "group": "Activity Pattern",
    },
]

def predict_account(data_dict):
    row = {}
    for feat in features:
        value = data_dict.get(feat, 0)
        try:
            row[feat] = float(value)
        except (TypeError, ValueError):
            row[feat] = 0.0

    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    X_scaled = scaler.transform(X)

    lr_prob = float(lr.predict_proba(X_scaled)[0][1])
    rf_prob = float(rf.predict_proba(X)[0][1])
    gb_prob = float(gb.predict_proba(X)[0][1])

    ensemble_prob = 0.10 * lr_prob + 0.45 * rf_prob + 0.45 * gb_prob
    ensemble_pred = int(ensemble_prob >= 0.35)

    probs = {
        "Logistic Regression": lr_prob,
        "Random Forest": rf_prob,
        "Gradient Boosting": gb_prob,
        "Ensemble": ensemble_prob,
    }

    top_algo = max(probs, key=probs.get)

    if ensemble_prob >= 0.25:
        verdict = "FAKE ACCOUNT DETECTED"
        verdict_class = "bot"
    elif ensemble_prob >= 0.10:
        verdict = "SUSPICIOUS ACCOUNT"
        verdict_class = "suspicious"
    else:
        verdict = "LEGITIMATE ACCOUNT"
        verdict_class = "legit"

    return {
        "prediction": ensemble_pred,
        "bot_probability": round(ensemble_prob, 4),
        "confidence_percent": round(max(ensemble_prob, 1 - ensemble_prob) * 100, 1),
        "verdict": verdict,
        "verdict_class": verdict_class,
        "top_algorithm": top_algo,
        "best_single_model": best_single,
        "model_probs": {k: round(v, 4) for k, v in probs.items()},
    }

@app.route("/")
def home():
    return render_template(
        "index.html",
        ui_fields=UI_FIELDS,
        features=features,
        best_single=best_single,
    )

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        incoming = request.get_json()
        data = incoming.get("features", {})
    else:
        data = request.form.to_dict()

    result = predict_account(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)   