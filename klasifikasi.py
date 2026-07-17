# ============================================================
#   KLASIFIKASI KECANDUAN GAME 
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ============================================================
# LOAD & PREPROCESSING 
# ============================================================

df = pd.read_csv('online_gaming_behavior.csv')

df['addicted'] = df['EngagementLevel'].apply(
    lambda x: 1 if x == 'High' else 0
)

X = df[['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes']]
y = df['addicted']

# Tambah random_state=42 supaya hasil konsisten tiap run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data training : {len(X_train)} baris")
print(f"Data testing  : {len(X_test)} baris\n")


# ============================================================
# TRAINING KEDUA MODEL
# ============================================================

# Decision Tree 
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# ============================================================
# EVALUASI KEDUA MODEL
# ============================================================

print("=" * 55)
print("   EVALUASI: DECISION TREE")
print("=" * 55)
print(classification_report(y_test, y_pred_dt,
      target_names=["Tidak Kecanduan (0)", "Kecanduan (1)"]))

print("=" * 55)
print("   EVALUASI: RANDOM FOREST")
print("=" * 55)
print(classification_report(y_test, y_pred_rf,
      target_names=["Tidak Kecanduan (0)", "Kecanduan (1)"]))


# ============================================================
# KUMPULKAN METRIK
# ============================================================

def get_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "Accuracy"  : accuracy_score(y_true, y_pred),
        "Precision" : report["weighted avg"]["precision"],
        "Recall"    : report["weighted avg"]["recall"],
        "F1-Score"  : report["weighted avg"]["f1-score"],
    }

dt_metrics = get_metrics(y_test, y_pred_dt)
rf_metrics = get_metrics(y_test, y_pred_rf)

print("Ringkasan Metrik:")
print(f"{'Metrik':<12} {'Decision Tree':>15} {'Random Forest':>15}")
print("-" * 44)
for m in dt_metrics:
    print(f"{m:<12} {dt_metrics[m]:>15.4f} {rf_metrics[m]:>15.4f}")


# ============================================================
# VISUALISASI BAR CHART
# ============================================================

PALETTE    = ["#5C6BC0", "#26A69A"]
BG_COLOR   = "#F8F9FA"
GRID_COLOR = "#DEE2E6"
TEXT_COLOR = "#212529"

metrics = list(dt_metrics.keys())
dt_vals = [dt_metrics[m] for m in metrics]
rf_vals = [rf_metrics[m] for m in metrics]

x     = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

bars_dt = ax.bar(x - width/2, dt_vals, width,
                 label="Decision Tree", color=PALETTE[0],
                 edgecolor="white", linewidth=1.2, zorder=3)
bars_rf = ax.bar(x + width/2, rf_vals, width,
                 label="Random Forest", color=PALETTE[1],
                 edgecolor="white", linewidth=1.2, zorder=3)

# Nilai di atas batang
for bar, color in zip(bars_dt, [PALETTE[0]] * len(bars_dt)):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=PALETTE[0])

for bar in bars_rf:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=PALETTE[1])

ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color=GRID_COLOR, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel("Metrik Evaluasi", fontsize=12, color=TEXT_COLOR, labelpad=10)
ax.set_ylabel("Nilai", fontsize=12, color=TEXT_COLOR, labelpad=10)
ax.set_title("Perbandingan Performa\nDecision Tree vs Random Forest",
             fontsize=15, fontweight="bold", color=TEXT_COLOR, pad=18)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, color=TEXT_COLOR)
ax.set_ylim(0, 1.12)
ax.tick_params(axis="y", colors=TEXT_COLOR)
ax.tick_params(axis="x", bottom=False)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID_COLOR)

legend_handles = [
    mpatches.Patch(facecolor=PALETTE[0], label="Decision Tree"),
    mpatches.Patch(facecolor=PALETTE[1], label="Random Forest"),
]
ax.legend(handles=legend_handles, fontsize=11,
          framealpha=0.85, edgecolor=GRID_COLOR, loc="upper right")

plt.tight_layout()
plt.savefig("perbandingan_model.png", dpi=150,
            bbox_inches="tight", facecolor=BG_COLOR)
plt.show()
print("Grafik disimpan: perbandingan_model.png")


# ============================================================
# INSIGHT ANALISIS
# ============================================================

best  = "Random Forest" if rf_metrics["F1-Score"] >= dt_metrics["F1-Score"] \
        else "Decision Tree"
diff  = abs(rf_metrics["F1-Score"] - dt_metrics["F1-Score"])

print("\n" + "=" * 55)
print("   INSIGHT ANALISIS")
print("=" * 55)
print(f"""
Model Terbaik : {best}
Selisih F1    : {diff:.4f}

Interpretasi Metrik:
• Accuracy   – {dt_metrics['Accuracy']:.3f} (DT) vs {rf_metrics['Accuracy']:.3f} (RF)
  Persentase prediksi benar secara keseluruhan.

• Precision  – {dt_metrics['Precision']:.3f} (DT) vs {rf_metrics['Precision']:.3f} (RF)
  Dari semua yang diprediksi kecanduan, berapa yang benar.
  Tinggi = sedikit false positive.

• Recall     – {dt_metrics['Recall']:.3f} (DT) vs {rf_metrics['Recall']:.3f} (RF)
  Dari semua yang benar kecanduan, berapa yang berhasil terdeteksi.
  Ini metrik PALING PENTING untuk deteksi kecanduan — kita tidak
  mau kasus kecanduan terlewat (false negative).

• F1-Score   – {dt_metrics['F1-Score']:.3f} (DT) vs {rf_metrics['F1-Score']:.3f} (RF)
  Rata-rata harmonis Precision & Recall. Metrik utama perbandingan.

Kesimpulan:
Random Forest unggul karena menggabungkan banyak Decision Tree
(ensemble), sehingga lebih tahan terhadap overfitting. Decision
Tree tunggal tanpa pruning cenderung "hafal" data training dan
performanya turun di data testing.
""")