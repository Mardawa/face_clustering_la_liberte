import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import pandas as pd
import pathlib
from tqdm import tqdm
from sklearn import tree
import os

# OpenCV Fast Fourier Transform (FFT) for blur detection in images
# Code taken from
# https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/


def detect_blur_fft(image, size=60, thresh=10, vis=False):
    """
    @image: input image for blur detection (BGR numpy array)
    @size: size of the radius around the centerpoint of the image for which we will zero out the FFT shift
    @thresh: value which the mean value of the magnitudes (more on that later) will be compared to for determining whether an image is considered blurry or not blurry
    @vis: boolean indicating whether to visualize/plot the original input image and magnitude image using matplotlib
    """

    # resize the image and convert to grayscale
    image = imutils.resize(image, width=500)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))
        # display the original input image
        (fig, ax) = plt.subplots(
            1,
            2,
        )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        # ax[0].set_xticks([])
        # ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        # ax[1].set_xticks([])
        # ax[1].set_yticks([])
        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


def detect_blur_laplacian(img, threshold=100, rescale=True):
    """
    @img: input image for blur detection (BGR numpy array)
    @threshold: value which the mean value of the laplacian will be compared to for determining whether an image is considered blurry or not blurry
    """
    if rescale:
        img = imutils.resize(img, width=500)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return (fm, fm < threshold)


def compare_blur_detection(img_path: str, plot=True):
    img = cv2.imread(img_path)  # cv2 load image as BGR

    ftt_threshold = 15.579238414764404
    laplacian_threshold = 165.1521759033203

    mean, blur_ftt = detect_blur_fft(img, size=60, thresh=ftt_threshold, vis=False)
    # fm, blur_laplacian = detect_blur_laplacian(img, threshold=165.1521759033203, rescale=False)
    fm_rescaled, blur_laplacian_rescaled = detect_blur_laplacian(img, threshold=laplacian_threshold)

    if plot:

        # plt subplots 1x3
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # plot original image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[0].imshow(img_rgb)
        ax[0].set_title("Original Image")

        # plot rescalled image
        img_rescaled = imutils.resize(img, width=500)
        gray = cv2.cvtColor(img_rescaled, cv2.COLOR_BGR2GRAY)
        ax[1].imshow(gray, cmap="gray")
        ax[1].set_title("Rescaled gray Image")

        def is_blurry(fft, laplacian_rescaled):
            if (fft > ftt_threshold) or (laplacian_rescaled > laplacian_threshold):
                return "Not blurry"
            elif (fft > 10) and (laplacian_rescaled > 100):
                return "Not Blurry"
            else:
                return "blurry"

        # plot text with blur detection results
        text = f"""
        FFT: {mean:.3f}
        Laplacian: {fm_rescaled:.3f}
        Decision: {is_blurry(mean, fm_rescaled)}
        """
        ax[2].text(0, 0.5, text, fontsize=12, ha="left", va="center")
        ax[2].set_title("Blur Detection Results")
        ax[2].axis("off")


def compare_blur_detection_quality(img_name: str, plot=True):

    img_original = cv2.imread(f"dataset/fullsize/faces/{img_name}")
    img_original_size = os.path.getsize(f"dataset/fullsize/faces/{img_name}")

    img_preview = cv2.imread(f"dataset/preview/faces/{img_name}")
    img_preview_size = os.path.getsize(f"dataset/preview/faces/{img_name}")

    ftt_threshold = 15.579238414764404
    laplacian_threshold = 165.1521759033203

    mean_original, _ = detect_blur_fft(img_original, size=60, thresh=ftt_threshold, vis=False)
    mean_preview, _ = detect_blur_fft(img_preview, size=60, thresh=ftt_threshold, vis=False)

    fm_rescaled_original, _ = detect_blur_laplacian(img_original, threshold=laplacian_threshold)
    fm_rescaled_preview, _ = detect_blur_laplacian(img_preview, threshold=laplacian_threshold)

    if plot:

        # plt subplots 1x3
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # plot original image
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        ax[0].imshow(img_original_rgb)
        # get image size
        height, width, _ = img_original.shape
        ax[0].set_title(f"Original Image ({width}x{height}) - {img_original_size/1024:.2f} KB")

        # plot preview image
        img_preview_rgb = cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB)
        ax[1].imshow(img_preview_rgb)
        # get image size
        height, width, _ = img_preview.shape
        ax[1].set_title(f"Preview Image ({width}x{height}) - {img_preview_size / 1024:.2f} KB")

        def is_blurry(fft, laplacian_rescaled):
            if (fft > ftt_threshold) or (laplacian_rescaled > laplacian_threshold):
                return "Not blurry"
            elif (fft > 10) and (laplacian_rescaled > 100):
                return "Not Blurry"
            else:
                return "blurry"

        # plot text with blur detection results
        text = f"""
        Original  
        FFT:        {mean_original:.3f} 
        Laplacian:  {fm_rescaled_original:.3f} 
        Decision:   {is_blurry(mean_original, fm_rescaled_original)} 

        Preview
        FFT:        {mean_preview:.3f}
        Laplacian:  {fm_rescaled_preview:.3f}
        Decision:   {is_blurry(mean_preview, fm_rescaled_preview)}
        """
        ax[2].text(0, 0.5, text, fontsize=12, ha="left", va="center")
        ax[2].set_title("Blur Detection Results")
        ax[2].axis("off")
