import re
import cv2
import pytesseract
import numpy as np

# to set relevant segmentation mode for pytesseract to recognize more text
custom_oem_psm_config = r'--oem 3 --psm 6'


def correct_rotation(image):
    # convert the image to grayscale and flip the foreground and background 
    # to ensure foreground is now "white" and the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to 255 
    # and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that are 
    # greater than zero, then use these coordinates to compute a rotated 
    # bounding box that contains all coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the range [-90, 0); 
    # as the rectangle rotates clockwise the returned angle trends to 0 
    # -- in this special case we need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def process_image(image):
    # use opencv de-noising module and apply on each image
    denoised_img = cv2.fastNlMeansDenoisingColored(image)

    # rotation correction
    image = correct_rotation(denoised_img)
    return image


def run_pytesseract_ocr(image):
    # convert image from BGR to RGB to pass to pytesseract
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img_rgb,
                                       config=custom_oem_psm_config)
    return text


def find_weight_and_height():
    pass


def predict(image_file_path):
    img = cv2.imread(image_file_path)

    img = process_image(img)
    text = run_pytesseract_ocr(img)

    print()
    print("PyTesseract result")
    print("***")
    print(text)
    print("***")
    print()

    return 0.00
