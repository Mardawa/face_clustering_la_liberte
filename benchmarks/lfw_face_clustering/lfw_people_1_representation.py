from deepface import DeepFace
from deepface.basemodels import VGGFace, Facenet, OpenFace, FbDeepFace, ArcFace, Facenet512, DeepID, DlibWrapper
from deepface.basemodels import SFaceWrapper

from tqdm.autonotebook import tqdm

import pandas as pd
import pathlib
import cv2

from sklearn.datasets import fetch_lfw_people

res_folder = pathlib.Path("lfw_people_benchmark")

# load lfw people
lfw_people = fetch_lfw_people(resize=1, color=True)

X = lfw_people["images"]
y = lfw_people["target"]

# put X and y in a dataframe
df = pd.DataFrame({"img_rgb": list(X), "target": y})

# convert rgb to bgr
# from deepface.represent
# img_path: exact image path, numpy array (BGR) or based64 encoded images could be passed.
df["img_bgr"] = df["img_rgb"].apply(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR))

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

# loop over models
for model_name, model in tqdm(models.items()):

    print(f"model: {model_name}")

    # load model
    model = model()

    # create a folder for the model
    model_folder = res_folder / "representation" / model_name
    model_folder.mkdir(exist_ok=True, parents=True)

    # apply the model to the images
    df[f"{model_name}_representation"] = df["img_bgr"].apply(
        lambda x: DeepFace.represent(
            img_path=x, model_name=model_name, model=model, enforce_detection=False, detector_backend="skip"
        )
    )

    # save the dataframe (pickle)
    df[["img_rgb", "target", f"{model_name}_representation"]].to_pickle(
        model_folder / f"{model_name}_representations.pkl"
    )

    # drop the representation column
    df.drop(columns=[f"{model_name}_representation"], inplace=True)

    # break # for testing
