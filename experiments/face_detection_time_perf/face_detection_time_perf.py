from deepface.detectors import (
    OpenCvWrapper,
    SsdWrapper,
    DlibWrapper,
    MtcnnWrapper,
    RetinaFaceWrapper,
    MediapipeWrapper,
    FaceDetector,
)

from time import perf_counter
import pathlib
from tqdm import tqdm
import cv2
import json


def detect_face_in_folder(src_folder_str: str, detect_face, face_detector) -> dict:
    """
    Detect faces of images in src_folder and save the faces in dst_folder

    @param src_folder: folder containing the images
    @param dst_folder: folder where the faces will be saved

    @return: dict of the form {
        "no_face": no_face, # number of images with no face detected
        "not_image": not_image, # number of files that could not be read
        "total_faces": total_faces, # total number of faces detected
        "confidences": confidences, # dictionnary of the confidences score of each detected face
        }
    """

    src_folder = pathlib.Path(src_folder_str)
    detection_time = 0
    nb_faces = 0

    for img_path in tqdm(list(src_folder.glob("*"))[:1000]):

        if img_path.suffix not in [".png", ".jpg"]:
            continue

        img_bgr = cv2.imread(str(img_path))  # retinface expects BGR, not RGB ! when given a str it use imread !

        if img_bgr is None:
            continue

        t0 = perf_counter()
        faces = detect_face(face_detector, img_bgr, align=False)
        t1 = perf_counter()
        nb_faces += len(faces)
        detection_time += t1 - t0

    return detection_time, nb_faces


def main():
    backends = {
        # "opencv": OpenCvWrapper.detect_face,
        # "ssd": SsdWrapper.detect_face,
        # "dlib": DlibWrapper.detect_face,
        # "mtcnn": MtcnnWrapper.detect_face,
        "retinaface": RetinaFaceWrapper.detect_face,
        # "mediapipe": MediapipeWrapper.detect_face,
    }

    src_folder_str = "dataset/img_fullsize"

    res = {}

    for detector_backend in backends:
        print(f"Testing {detector_backend}...")

        detect_face = backends.get(detector_backend)
        face_detector = FaceDetector.build_model(detector_backend)

        t_start = perf_counter()  # time excluding the building time of the model
        detection_time, nb_faces = detect_face_in_folder(src_folder_str, detect_face, face_detector)
        t_end = perf_counter()
        total_time = t_end - t_start

        res[detector_backend] = {
            "detection_time": detection_time,
            "nb_faces": nb_faces,
            "total_time": total_time,
        }

        # save res
        with open("face_detection_time_perf.csv", "a") as fp:
            # write the results
            fp.write(f"{detector_backend};{detection_time};{nb_faces};{total_time}\n")


# if main
if __name__ == "__main__":
    main()
