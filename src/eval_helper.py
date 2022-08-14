import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(aspect_preds, aspect_labels, polarity_preds, polarity_labels):
    assert len(aspect_labels) == len(aspect_preds) == len(polarity_labels) == len(polarity_preds)
    aspect_transpose_preds = np.transpose(aspect_preds)
    aspect_transpose_labels = np.transpose(aspect_labels)

    polarity_transpose_preds = np.transpose(polarity_preds)
    polarity_transpose_labels = np.transpose(polarity_labels)

    f1_scores = []
    r2_scores = [1 for _ in range(aspect_transpose_labels.shape[0])]
    max_sentiment = 5
    min_sentiment = 1
    for i in range(aspect_transpose_preds.shape[0]):
        f1_scores.append(f1_score(aspect_transpose_labels[i], aspect_transpose_preds[i]))

    for i in range(polarity_transpose_preds.shape[0]):
        if f1_scores[i] == 0:
            r2_scores[i] = 1
        else:
            a = 0
            b = 0
            for j in range(polarity_transpose_preds.shape[1]):
                if aspect_transpose_preds[i][j] != 0 and aspect_transpose_labels[i][j] != 0:
                    a += (polarity_transpose_preds[i][j] - polarity_transpose_labels[i][j]) ** 2
                    b += (max_sentiment - min_sentiment) ** 2

            r2_scores[i] = 1 - a / b

    score = 0
    for i in range(len(f1_scores)):
        score += f1_scores[i] * r2_scores[i]

    score /= len(f1_scores)

    aspect_f1 = f1_score(aspect_labels.flatten(), aspect_preds.flatten())
    polarity_f1 = f1_score(polarity_labels.flatten(), polarity_preds.flatten(), average='weighted')
    results = {
        "aspect_f1": aspect_f1,
        "polarity_f1": polarity_f1,
        "mean_aspect_polarity": (aspect_f1 + polarity_f1) / 2,
        "competition_score": score,
    }
    return results
