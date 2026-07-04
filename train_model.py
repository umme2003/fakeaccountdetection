"""
train_model.py
--------------
Full ML pipeline for Fake Account & Bot Detection.

Models trained:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. Gradient Boosting (sklearn, equivalent to XGBoost)
  4. Voting Ensemble (soft vote)

Steps:
  - Load & inspect dataset
  - EDA with feature distributions
  - Preprocessing (scaling, encoding)
  - Train/Val/Test split (70/15/15), stratified
  - Train all models with 5-fold cross-validation
  - Evaluate: accuracy, precision, recall, F1, ROC-AUC
  - Confusion matrix & ROC curves
  - Feature importance (RF + GB)
  - Save best model as bot_detector.pkl
  - Save full report
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# import dataset generator
import sys
sys.path.insert(0, '/home/claude/bot_detection')
from generate_dataset import generate_dataset

OUT = '/home/claude/bot_detection'
PLOTS = f'{OUT}/plots'
import os; os.makedirs(PLOTS, exist_ok=True)

# ─── PALETTE ─────────────────────────────────────────────────────────────────
DARK   = '#0a0b0f'
GREEN  = '#00e676'
RED    = '#ff3c6e'
BLUE   = '#3c8eff'
YELLOW = '#ffaa00'
MUTED  = '#64748b'
TEXT   = '#e2e8f0'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': '#111318',
    'axes.edgecolor': '#1e2430', 'grid.color': '#1e2430',
    'text.color': TEXT, 'axes.labelcolor': TEXT,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.titlecolor': TEXT, 'axes.titlesize': 12,
    'font.family': 'monospace', 'axes.grid': True,
    'legend.facecolor': '#111318', 'legend.edgecolor': '#1e2430',
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GENERATE & LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  BOTSENTRY ML — FAKE ACCOUNT DETECTION PIPELINE")
print("=" * 60)
print("\n[1/7] Generating synthetic dataset...")

df, bots_meta = generate_dataset()
df.to_csv(f'{OUT}/dataset.csv', index=False)

FEATURES = [c for c in df.columns if c != 'label']
X = df[FEATURES].values
y = df['label'].values

print(f"  Total samples : {len(df):,}")
print(f"  Legitimate    : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"  Bot/Fake      : {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
print(f"  Features      : {len(FEATURES)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Generating EDA plots...")

KEY_FEATURES = [
    'ff_ratio', 'burst_score', 'name_entropy', 'account_age_days',
    'spam_keyword_score', 'duplicate_post_ratio', 'avg_daily_posts',
    'mutual_follow_rate', 'content_diversity', 'posting_hour_std',
    'sentiment_variance', 'avg_hashtags_per_post'
]

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
fig.patch.set_facecolor(DARK)
fig.suptitle('Feature Distributions: Legitimate vs Bot Accounts', fontsize=14, y=0.98)

df_legit = df[df['label'] == 0]
df_bot   = df[df['label'] == 1]

for idx, feat in enumerate(KEY_FEATURES):
    ax = axes[idx // 3][idx % 3]
    ax.hist(df_legit[feat], bins=40, alpha=0.7, color=GREEN,  label='Legitimate', density=True)
    ax.hist(df_bot[feat],   bins=40, alpha=0.7, color=RED,    label='Bot/Fake',   density=True)
    ax.set_title(feat.replace('_', ' ').title(), fontsize=10)
    ax.set_xlabel('Value', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{PLOTS}/01_feature_distributions.png', dpi=120, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: 01_feature_distributions.png")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(16, 13))
fig.patch.set_facecolor(DARK)
corr = df[FEATURES + ['label']].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 130, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.3, linecolor='#1e2430',
            annot=False, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=13)
plt.tight_layout()
plt.savefig(f'{PLOTS}/02_correlation_heatmap.png', dpi=120, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: 02_correlation_heatmap.png")

# Class balance bar
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor(DARK)
counts = [(y==0).sum(), (y==1).sum()]
bars = ax.bar(['Legitimate', 'Bot/Fake'], counts, color=[GREEN, RED], width=0.5, edgecolor='none')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{count:,}\n({count/len(y)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
ax.set_title('Class Distribution', fontsize=12)
ax.set_ylabel('Number of Accounts')
ax.set_ylim(0, max(counts) * 1.2)
plt.tight_layout()
plt.savefig(f'{PLOTS}/03_class_distribution.png', dpi=120, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: 03_class_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Splitting data (70/15/15 stratified)...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)  # 0.1765 * 0.85 ≈ 0.15

print(f"  Train : {len(X_train):,}  Val : {len(X_val):,}  Test : {len(X_test):,}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DEFINE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Training models with 5-fold cross-validation...")

models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=15,
                                                   min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                       max_depth=5, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    X_use = X_train_s if name == 'Logistic Regression' else X_train
    scores = cross_val_score(model, X_use, y_train, cv=cv, scoring='f1', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<22} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

# ─── FIT ALL MODELS ──────────────────────────────────────────────────────────
print("\n  Fitting final models on full training set...")
fitted = {}

lr = LogisticRegression(
    C=1.0, 
    max_iter=1000, 
    class_weight={0: 1, 1: 3}, 
    random_state=42
)
lr.fit(X_train_s, y_train)
fitted['Logistic Regression'] = lr

rf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=18, 
    min_samples_leaf=1, 
    class_weight={0: 1, 1: 3}, 
    random_state=42, 
    n_jobs=-1
)
rf.fit(X_train, y_train)
fitted['Random Forest'] = rf

gb = GradientBoostingClassifier(
    n_estimators=150, 
    learning_rate=0.08, 
    max_depth=5, 
    random_state=42
)
gb.fit(X_train, y_train)
fitted['Gradient Boosting'] = gb

# Voting ensemble (uses unscaled for RF & GB, scaled for LR)
# We'll evaluate ensemble manually with soft probabilities
def ensemble_predict_proba(X_s, X):
    p_lr = lr.predict_proba(X_s)
    p_rf = rf.predict_proba(X)
    p_gb = gb.predict_proba(X)
    return 0.25 * p_lr + 0.40 * p_rf + 0.35 * p_gb

def ensemble_predict(X_s, X, threshold=0.5):
    proba = ensemble_predict_proba(X_s, X)
    return (proba[:, 1] >= threshold).astype(int)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Evaluating on held-out test set...")
print(f"\n{'Model':<24} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7}")
print("-" * 60)

all_results = {}
for name, model in fitted.items():
    if name == 'Logistic Regression':
        y_pred  = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
    else:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    all_results[name] = dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                              y_pred=y_pred, y_proba=y_proba, cm=cm)
    print(f"{name:<24} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} {f1:>6.4f} {auc:>7.4f}")

# Ensemble
ens_pred  = ensemble_predict(X_test_s, X_test)
ens_proba = ensemble_predict_proba(X_test_s, X_test)[:, 1]
acc  = accuracy_score(y_test, ens_pred)
prec = precision_score(y_test, ens_pred)
rec  = recall_score(y_test, ens_pred)
f1   = f1_score(y_test, ens_pred)
auc  = roc_auc_score(y_test, ens_proba)
cm   = confusion_matrix(y_test, ens_pred)
all_results['Ensemble (Soft Vote)'] = dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                                            y_pred=ens_pred, y_proba=ens_proba, cm=cm)
print(f"{'Ensemble (Soft Vote)':<24} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} {f1:>6.4f} {auc:>7.4f}")
print()
print(f"\nBest Model Classification Report (Ensemble):")
print(classification_report(y_test, ens_pred, target_names=['Legitimate', 'Bot/Fake']))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PLOTS — Metrics, ROC, Confusion Matrix, Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
print("[6/7] Generating evaluation plots...")

# ── Metrics comparison bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(DARK)
metric_names = ['acc', 'prec', 'rec', 'f1', 'auc']
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
model_names = list(all_results.keys())
colors = [BLUE, GREEN, RED, YELLOW]
x = np.arange(len(labels))
width = 0.2

for i, (mname, col) in enumerate(zip(model_names, colors)):
    vals = [all_results[mname][m] for m in metric_names]
    bars = ax.bar(x + i * width, vals, width, label=mname, color=col, alpha=0.85)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(labels)
ax.set_ylim(0.75, 1.02)
ax.set_title('Model Performance Comparison (Test Set)', fontsize=12)
ax.set_ylabel('Score')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{PLOTS}/04_model_comparison.png', dpi=120, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: 04_model_comparison.png")

# ── ROC Curves ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(DARK)
roc_colors = [BLUE, GREEN, RED, YELLOW]
for (mname, res), col in zip(all_results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{mname} (AUC={res['auc']:.4f})")
ax.plot([0,1],[0,1], '--', color=MUTED, lw=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{PLOTS}/05_roc_curves.png', dpi=120, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: 05_roc_curves.png")

# ── Confusion Matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle('Confusion Matrices (Test Set)', fontsize=13)
cmap_custom = sns.light_palette(GREEN, as_cmap=True)

for ax, (mname, res) in zip(axes, all_results.items()):
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap=cmap_custom, ax=ax,
                linewidths=1, linecolor='#1e2430',
                xticklabels=['Legit','Bot'], yticklabels=['Legit','Bot'])
    ax.set_title(mname, fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{PLOTS}/06_confusion_matrices.png', dpi=120, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: 06_confusion_matrices.png")

# ── Feature Importance ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle('Feature Importance', fontsize=13)

for ax, (mname, model) in zip(axes, [('Random Forest', rf), ('Gradient Boosting', gb)]):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # top 20
    feat_names = [FEATURES[i] for i in indices]
    vals = importances[indices]

    colors_bar = [GREEN if v > np.percentile(vals, 70) else BLUE if v > np.percentile(vals, 40) else MUTED
                  for v in vals]
    ax.barh(range(len(indices)), vals, color=colors_bar, edgecolor='none')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f.replace('_',' ') for f in feat_names], fontsize=8)
    ax.set_title(mname, fontsize=11)
    ax.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig(f'{PLOTS}/07_feature_importance.png', dpi=120, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: 07_feature_importance.png")

# ── CV Score distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK)
cv_models = list(cv_results.keys())
cv_data   = [cv_results[m] for m in cv_models]
bp = ax.boxplot(cv_data, patch_artist=True, medianprops=dict(color=DARK, lw=2))
box_colors = [BLUE, GREEN, RED]
for patch, col in zip(bp['boxes'], box_colors):
    patch.set_facecolor(col); patch.set_alpha(0.7)
for whisker in bp['whiskers']: whisker.set_color(MUTED)
for cap in bp['caps']: cap.set_color(MUTED)
ax.set_xticklabels([m.replace(' ','\n') for m in cv_models])
ax.set_title('5-Fold Cross-Validation F1 Scores', fontsize=12)
ax.set_ylabel('F1 Score')
plt.tight_layout()
plt.savefig(f'{PLOTS}/08_cv_scores.png', dpi=120, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: 08_cv_scores.png")

# ── Probability distribution of ensemble predictions ────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(DARK)
legit_probs = ens_proba[y_test == 0]
bot_probs   = ens_proba[y_test == 1]
ax.hist(legit_probs, bins=50, alpha=0.75, color=GREEN, label='Legitimate', density=True)
ax.hist(bot_probs,   bins=50, alpha=0.75, color=RED,   label='Bot/Fake',   density=True)
ax.axvline(0.5, color=YELLOW, linestyle='--', lw=1.5, label='Decision threshold')
ax.set_xlabel('Predicted Bot Probability')
ax.set_ylabel('Density')
ax.set_title('Ensemble: Predicted Probability Distribution', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(f'{PLOTS}/09_probability_distribution.png', dpi=120, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  Saved: 09_probability_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE MODEL & REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Saving model and report...")

# Best single model = Random Forest (usually highest or tied with GB)
best_model_name = max(
    ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    key=lambda m: all_results[m]['f1']
)
print(f"  Best single model: {best_model_name}")

model_bundle = {
    'scaler':      scaler,
    'rf':          rf,
    'gb':          gb,
    'lr':          lr,
    'features':    FEATURES,
    'best_single': best_model_name,
}
with open(f'{OUT}/bot_detector.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)
print("  Saved: bot_detector.pkl")

# ── Text report ───────────────────────────────────────────────────────────────
report_lines = [
    "=" * 60,
    "  BOTSENTRY ML — MODEL EVALUATION REPORT",
    "=" * 60,
    "",
    f"Dataset: {len(df):,} accounts  |  {(y==0).sum():,} Legitimate  |  {(y==1).sum():,} Bots",
    f"Features: {len(FEATURES)}  |  Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}",
    "",
    f"{'Model':<24} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>8}",
    "-" * 60,
]
for mname, res in all_results.items():
    report_lines.append(
        f"{mname:<24} {res['acc']:>7.4f} {res['prec']:>7.4f} {res['rec']:>7.4f} "
        f"{res['f1']:>7.4f} {res['auc']:>8.4f}"
    )

report_lines += [
    "",
    "Best Single Model: " + best_model_name,
    "",
    "Detailed Classification Report (Ensemble):",
    classification_report(y_test, ens_pred, target_names=['Legitimate', 'Bot/Fake']),
    "",
    "Top 10 Most Important Features (Random Forest):",
]
fi_idx = np.argsort(rf.feature_importances_)[::-1][:10]
for rank, idx in enumerate(fi_idx, 1):
    report_lines.append(f"  {rank:2}. {FEATURES[idx]:<32} {rf.feature_importances_[idx]:.4f}")

report_lines += ["", "=" * 60]
report_text = "\n".join(report_lines)
print(report_text)
with open(f'{OUT}/evaluation_report.txt', 'w') as f:
    f.write(report_text)
print("\n  Saved: evaluation_report.txt")
print("\n✓ Pipeline complete. All outputs saved to:", OUT)
