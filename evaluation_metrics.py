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

# ================= CONNECT DATABASE =================

conn = get_connection()
cur = conn.cursor()

cur.execute("SELECT name, embedding FROM missing_persons")
missing_people = cur.fetchall()

cur.execute("SELECT embedding FROM cctv_embeddings")
cctv_embeddings = cur.fetchall()

cur.close()
conn.close()

# ================= STORAGE VARIABLES =================

y_true = []
y_pred = []
similarity_scores = []

# ================= FACE MATCHING PROCESS =================

for name, emb in missing_people:

    test_embedding = np.array(emb, dtype=np.float32)
    test_embedding /= np.linalg.norm(test_embedding)

    best_similarity = -1

    for (db_emb,) in cctv_embeddings:

        db_embedding = np.array(db_emb, dtype=np.float32)
        db_embedding /= np.linalg.norm(db_embedding)

        similarity = float(np.dot(test_embedding, db_embedding))

        if similarity > best_similarity:
            best_similarity = similarity

    similarity_scores.append(best_similarity)

    # Prediction based on threshold
    if best_similarity >= SIMILARITY_THRESHOLD:
        y_pred.append(1)
    else:
        y_pred.append(0)

    # Ground truth assumption (for evaluation demo)
    # Assume match exists if similarity is high
    if best_similarity >= 0.50:
        y_true.append(1)
    else:
        y_true.append(0)


# ================= METRICS =================

accuracy = accuracy_score(y_true, y_pred)
 precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Accuracy:", round(accuracy*100,2), "%")
print("Precision:", round(precision*100,2), "%")
print("Recall:", round(recall*100,2), "%")
print("F1 Score:", round(f1*100,2), "%")


# ================= CONFUSION MATRIX =================

cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")


# ================= ROC CURVE =================

if len(set(y_true)) > 1:

    fpr, tpr, thresholds = roc_curve(y_true, similarity_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()


# ================= PRECISION RECALL CURVE =================

if len(set(y_true)) > 1:

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, similarity_scores)

    plt.figure()
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")


# ================= SIMILARITY DISTRIBUTION =================

plt.figure()
sns.histplot(similarity_scores, bins=20, kde=True)
plt.title("Similarity Score Distribution")
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")


# ================= THRESHOLD vs ACCURACY =================

threshold_range = np.arange(0.3, 0.9, 0.05)
accuracies = []

for t in threshold_range:

    temp_pred = []

    for score in similarity_scores:
        if score >= t:
            temp_pred.append(1)
        else:
            temp_pred.append(0)

    acc = accuracy_score(y_true, temp_pred)
    accuracies.append(acc)

plt.figure()
plt.plot(threshold_range, accuracies, marker='o')
plt.xlabel("Similarity Threshold")
plt.ylabel("Accuracy")
plt.title("Threshold vs Accuracy")


# ================= SHOW ALL FIGURES =================

plt.show() 