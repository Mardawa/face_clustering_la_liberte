from deepface.basemodels import VGGFace, Facenet, OpenFace, FbDeepFace, ArcFace, Facenet512, DeepID, DlibWrapper
from deepface.basemodels import SFaceWrapper

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import json

models = {
    "SFace": SFaceWrapper.load_model,
    "Facenet": Facenet.loadModel,
    "Facenet512": Facenet512.loadModel,
    "VGG-Face": VGGFace.loadModel,
    "OpenFace": OpenFace.loadModel,
    "DeepFace": FbDeepFace.loadModel,
    "DeepID": DeepID.loadModel,
    "Dlib": DlibWrapper.loadModel,
    "ArcFace": ArcFace.loadModel,
}

distance_metrics = [
    "cosine",
    "euclidean",
    "euclidean_l2",
]

res_accuracy_path = "results/accuracy"
pathlib.Path(res_accuracy_path).mkdir(parents=True, exist_ok=True)

res_df = pd.DataFrame(
    columns=[
        "model",
        "distance_metric",
        "accuracy",
        "accuracy_se",
        "precision",
        "precision_se",
        "recall",
        "recall_se",
        "f1",
        "f1_se",
    ]
)

for model_name in tqdm(models.keys()):

    img1_colname = f"{model_name}_img1"
    img2_colname = f"{model_name}_img2"

    df = pd.read_csv(
        f"results/distances/{model_name}_distances.csv",
        index_col=0,
        converters={img1_colname: json.loads, img2_colname: json.loads},
    )

    for distance_metric in distance_metrics:

        df[f"y_pred_{distance_metric}"] = df[f"y_pred_{distance_metric}"].astype(int)

        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for fold in range(10):
            df_fold = df[df["fold"] == fold]
            accuracy = accuracy_score(df_fold["y_true"], df_fold[f"y_pred_{distance_metric}"])
            accuracies.append(accuracy)

            precision = precision_score(df_fold["y_true"], df_fold[f"y_pred_{distance_metric}"])
            precisions.append(precision)

            recall = recall_score(df_fold["y_true"], df_fold[f"y_pred_{distance_metric}"])
            recalls.append(recall)

            f1 = f1_score(df_fold["y_true"], df_fold[f"y_pred_{distance_metric}"])
            f1s.append(f1)

        accuracy_mean = np.mean(accuracies)
        accuracy_se = np.std(accuracies, ddof=1) / np.sqrt(10)

        precision_mean = np.mean(precisions)
        precision_se = np.std(precisions, ddof=1) / np.sqrt(10)

        recall_mean = np.mean(recalls)
        recall_se = np.std(recalls, ddof=1) / np.sqrt(10)

        f1_mean = np.mean(f1s)
        f1_se = np.std(f1s, ddof=1) / np.sqrt(10)

        # concat the results
        res_df = pd.concat(
            [
                res_df,
                pd.DataFrame(
                    {
                        "model": model_name,
                        "distance_metric": distance_metric,
                        "accuracy": accuracy_mean,
                        "accuracy_se": accuracy_se,
                        "precision": precision_mean,
                        "precision_se": precision_se,
                        "recall": recall_mean,
                        "recall_se": recall_se,
                        "f1": f1_mean,
                        "f1_se": f1_se,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

res_df.to_csv(f"{res_accuracy_path}/accuracy.csv")
