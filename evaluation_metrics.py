
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve
)

from database import get_connection

SIMILARITY_THRESHOLD = 0.40
GROUND_TRUTH_THRESHOLD = 0.50

# ================= CONNECT DATABASE =================

conn = get_connection()
cur = conn.cursor()

try:
    cur.execute("SELECT name, embedding FROM missing_persons")
    missing_people = cur.fetchall()

    cur.execute("SELECT embedding FROM cctv_embeddings")
    cctv_embeddings = cur.fetchall()

finally:
    cur.close()
    conn.close()

# ================= STORAGE VARIABLES =================

y_true = []
y_pred = []
similarity_scores = []

# ================= FACE MATCHING PROCESS =================

valid_cctv_embeddings = []

for (emb,) in cctv_embeddings:
    if emb is None:
        continue

    db_embedding = np.asarray(emb, dtype=np.float32).flatten()
    norm = np.linalg.norm(db_embedding)

    if norm == 0 or not np.isfinite(norm) or not np.all(np.isfinite(db_embedding)):
        continue

    valid_cctv_embeddings.append(db_embedding / norm)

if not valid_cctv_embeddings:
    raise ValueError("No valid CCTV embeddings found in the database.")

for name, emb in missing_people:

    if emb is None:
        continue

    test_embedding = np.asarray(emb, dtype=np.float32).flatten()
    norm = np.linalg.norm(test_embedding)

    if norm == 0 or not np.isfinite(norm) or not np.all(np.isfinite(test_embedding)):
        print(f"Skipping invalid embedding for: {name}")
        continue

    test_embedding /= norm

    best_similarity = -1

    for db_embedding in valid_cctv_embeddings:

        if test_embedding.shape != db_embedding.shape:
            continue

        similarity = float(np.dot(test_embedding, db_embedding))

        if similarity > best_similarity:
            best_similarity = similarity

    if best_similarity == -1:
        print(f"No compatible CCTV embedding found for: {name}")
        continue

    similarity_scores.append(best_similarity)

    # Prediction based on similarity threshold
    y_pred.append(
        1 if best_similarity >= SIMILARITY_THRESHOLD else 0
    )

    # Ground-truth assumption for evaluation demo only
    y_true.append(
        1 if best_similarity >= GROUND_TRUTH_THRESHOLD else 0
    )

# ================= VALIDATE DATA =================

if not y_true:
    raise ValueError("No valid face embeddings available for evaluation.")

# ================= METRICS =================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n========== EVALUATION RESULTS ==========")

print("Total Samples:", len(y_true))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

# ================= CONFUSION MATRIX =================

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# ================= ROC CURVE =================

if len(set(y_true)) > 1:

    fpr, tpr, thresholds = roc_curve(y_true, similarity_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

else:
    print("ROC curve skipped: only one class is present.")

# ================= PRECISION-RECALL CURVE =================

if len(set(y_true)) > 1:

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_true, similarity_scores
    )

    plt.figure()
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()

else:
    print("Precision-Recall curve skipped: only one class is present.")

# ================= SIMILARITY DISTRIBUTION =================

plt.figure()
sns.histplot(similarity_scores, bins=20, kde=True)
plt.title("Similarity Score Distribution")
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")
plt.tight_layout()

# ================= THRESHOLD VS ACCURACY =================

threshold_range = np.arange(0.3, 0.9, 0.05)
accuracies = []

for threshold in threshold_range:

    temp_pred = [
        1 if score >= threshold else 0
        for score in similarity_scores
    ]

    acc = accuracy_score(y_true, temp_pred)
    accuracies.append(acc)

plt.figure()
plt.plot(threshold_range, accuracies, marker="o")
plt.xlabel("Similarity Threshold")
plt.ylabel("Accuracy")
plt.title("Threshold vs Accuracy")
plt.tight_layout()

# ================= SHOW ALL FIGURES =================

plt.show()
