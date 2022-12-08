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
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pathlib


face_detector = FaceDetector.build_model("retinaface")


def retinaface_detect_face(img, align=True, threshold=0.9):
    """
    Modified version of RetinaFaceWrapper.detect_face:
        * threshold is now a parameter
        * include the confidence score in the response
    """

    from retinaface import RetinaFace
    from retinaface.commons import postprocess

    # ---------------------------------

    resp = []

    # The BGR2RGB conversion will be done in the preprocessing step of retinaface.
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #retinaface expects RGB but OpenCV read BGR

    """
    face = None
    img_region = [0, 0, img.shape[0], img.shape[1]] #Really?

    faces = RetinaFace.extract_faces(img_rgb, model = face_detector, align = align)

    if len(faces) > 0:
        face = faces[0][:, :, ::-1]

    return face, img_region
    """

    # --------------------------

    obj = RetinaFace.detect_faces(img, model=face_detector, threshold=threshold)

    if type(obj) == dict:
        for key in obj:
            identity = obj[key]
            facial_area = identity["facial_area"]
            confidence = identity["score"]

            y = facial_area[1]
            h = facial_area[3] - y
            x = facial_area[0]
            w = facial_area[2] - x
            img_region = [x, y, w, h]

            # detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
            detected_face = img[facial_area[1] : facial_area[3], facial_area[0] : facial_area[2]]

            if align:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                # mouth_right = landmarks["mouth_right"]
                # mouth_left = landmarks["mouth_left"]

                detected_face = postprocess.alignment_procedure(detected_face, right_eye, left_eye, nose)

            resp.append((detected_face, img_region, confidence))

    return resp


def retinaface_detect_face_in_folder(src_folder_str: str, dst_folder_str: str, threshold=0.90) -> dict:
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

    no_face = 0  # number of images without face detected
    not_image = 0  # all images that could not be read
    total_faces = 0  # total number of faces detected

    # confidence score for each face detected
    #   key:    image name
    #   value:  confidence scores
    confidences = {}

    src_folder = pathlib.Path(src_folder_str)
    src_folder.mkdir(parents=True, exist_ok=True)
    dst_folder = pathlib.Path(dst_folder_str)
    dst_folder.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(list(src_folder.glob("*"))):

        if img_path.suffix not in [".png", ".jpg"]:
            not_image += 1
            continue

        try:
            img_bgr = cv2.imread(str(img_path))  # retinface expects BGR, not RGB ! when given a str it use imread !
            # check if the image could be read
            if img_bgr is None:
                not_image += 1
                continue
            # img_rbg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except:
            not_image += 1
            continue

        faces = retinaface_detect_face(img_bgr, align=True, threshold=threshold)

        if len(faces) == 0:
            no_face += 1
            continue

        for idx, face in enumerate(faces):
            align_face = face[0]
            # img_region = face[1]
            confidence = face[2]

            align_face_name = f"{img_path.stem}_f{idx}{img_path.suffix}"

            # save the face
            # align_face = cv2.cvtColor(align_face, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(dst_folder / align_face_name), align_face)

            # save the confidence score
            confidences[align_face_name] = confidence
            total_faces += 1

    out = {
        "no_face": no_face,
        "not_image": not_image,
        "total_faces": total_faces,
        "confidences": confidences,
    }

    return out


backends = {
    "opencv": OpenCvWrapper.detect_face,
    "ssd": SsdWrapper.detect_face,
    "dlib": DlibWrapper.detect_face,
    "mtcnn": MtcnnWrapper.detect_face,
    "retinaface": RetinaFaceWrapper.detect_face,
    "mediapipe": MediapipeWrapper.detect_face,
}


def compare_face_detection(img_bgr, verbose=False) -> dict:
    """
    Compare the face dectection on a given image
    """
    # 3x2 grid of images
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))
    axs = axs.flatten()
    fig.tight_layout()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    model_perf = {}

    for i, detector_backend in enumerate(backends):

        if verbose:
            print("detector_backend: ", detector_backend, end="")

        detect_face = backends.get(detector_backend)
        face_detector = FaceDetector.build_model(detector_backend)

        t0 = perf_counter()  # time excluding the building time of the model
        obj = detect_face(face_detector, img_bgr, align=True)
        t1 = perf_counter()

        if verbose:
            print(" - time: ", t1 - t0)

        axs[i].imshow(img_rgb)
        axs[i].set_title(f"{detector_backend} - {t1 - t0:.3f}s")
        model_perf[detector_backend] = t1 - t0
        for face in obj:
            x, y, w, h = face[1]
            axs[i].add_patch(plt.Rectangle((x, y), w, h, fill=False, color="red", linewidth=2))

    return model_perf


def draw_bounding_boxes(img_bgr, faces, ax=None):
    """
    Draw the bounding boxes on the image
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_rgb)
    for face in faces:
        x, y, w, h = face[1]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color="red", linewidth=2))
        # confidence = face[2]
        # ax.text(x, y, f"{confidence:.2f}", color="red", fontsize=8)


def extract_face(img_path: pathlib.Path, save_folder: pathlib.Path, threshold: float = 0.95):
    img_bgr = cv2.imread(str(img_path))
    faces = retinaface_detect_face(img_bgr, align=True, threshold=threshold)
    if len(faces) == 0:
        return None

    for idx, face in enumerate(faces):
        align_face = face[0]
        # img_region = face[1]
        # confidence = face[2]

        align_face_name = f"{img_path.stem}_f{idx}{img_path.suffix}"

        # save the face
        # align_face = cv2.cvtColor(align_face, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(save_folder / align_face_name), align_face)
