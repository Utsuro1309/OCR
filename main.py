import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
from matplotlib import pyplot as plt
import enchant
import re
import tkinter as tk
from tkinter import filedialog

''''------------------------------------------------------Insert Image-----------------------------------------------'''
class ImageInsertion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("400x400")
        self.title("Insert Image")

        self.image_label = tk.Label(self)
        self.image_label.pack()

        self.insert_button = tk.Button(self, text="Insert your image", command=self.insert_image)
        self.insert_button.pack(pady=100)

        self.file_path = None

    def insert_image(self):
        # Open a file dialog to select an image file
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        # Load the image file and display it in the label
        if self.file_path:
            image = Image.open(self.file_path)

            w, h = self.winfo_width(), self.winfo_height()
            image = image.resize((w, h))

            photo_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo_image)
            self.image_label.image = photo_image

    def save_path(self):
        # Save the file path of the inserted image to a text file
        if self.file_path:
            with open("image_path.txt", "w") as file:
                file.write(self.file_path)


app = ImageInsertion()
app.mainloop()
app.save_path()

with open('image_path.txt', 'r') as file:
    path = file.read()

print(path)
'''--------------------------------------------------Display Image--------------------------------------------'''

def display(path):
    dpi = 80
    im_data = plt.imread(path)

    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    fig_size = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()


'''----------------------------------------------PreProcess--------------------------------------------------------'''

# PreProcess the original image (first layer)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Define the structuring element for the morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

# Perform an opening operation to remove small objects from the foreground
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imwrite("temp/opening.jpg", opening)
im = cv2.imread("temp/opening.jpg")

# Perform a closing operation to fill in small holes in the foreground
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# PreProcess the original image (second layer)

# im = cv2.imread(path)

# 1) Fix distortion in an image
def apply_distortion(img, k=0.1, p=0.1):
    distortion_coeffs = np.array([k, k, p, p], dtype=np.float32)

    img_size = (img.shape[1], img.shape[0])
    focal_length = img_size[1]
    camera_matrix = np.array([[focal_length, 0, img_size[0] / 2],
                              [0, focal_length, img_size[1] / 2],
                              [0, 0, 1]], dtype=np.float32)

    img_distorted = cv2.undistort(img, camera_matrix, distortion_coeffs)
    return img_distorted

img_distorted = apply_distortion(im, k=0.5, p=0.05)
cv2.imwrite('temp/page_dt.jpg', img_distorted)

def undistort_image(img, K, dist_coeffs):
    # Get the size of the input image
    img_size = (img.shape[1], img.shape[0])

    # Generate the optimal camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, img_size, 0)

    # Undistort the image
    undistorted_img = cv2.undistort(img, K, dist_coeffs, None, new_camera_matrix)

    return undistorted_img

# Load the input image
img = cv2.imread('temp/page_dt.jpg')

# Define the camera matrix and distortion coefficients
K = np.array([[1522, 0, 600],
              [0, 1522, 761],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.3, -0.7, -0.05, -0.05, 0], dtype=np.float32)

# Undistort the image
undistorted_img = undistort_image(img, K, dist_coeffs)
cv2.imwrite('temp/page_udt.jpg', undistorted_img)

# 2) Inverted Image
inverted_image = cv2.bitwise_not(im)
cv2.imwrite("temp/inverted.jpg", inverted_image)

# 3) Binarization
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(im)
cv2.imwrite("temp/gray.jpg", gray_image)

# im_bw = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,10)
thresh, im_bw = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
cv2.imwrite("temp/bw_image.jpg", im_bw)


# 4) Noise Removal
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)

    # main things that get rid of the noise
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 1)
    return image

no_noise = noise_removal(im_bw)
cv2.imwrite("temp/no_noise.jpg", no_noise)


# 5) Dilation(thicker) and Erosion(thinner)
def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

erode_image = thick_font(im_bw)
cv2.imwrite("temp/erode_image.jpg", erode_image)

# 6) Removing Borders (Must do before Deskew) (Use when text is all the way up agaisnt the border)
def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


no_borders = remove_borders(im_bw)
cv2.imwrite("temp/no_borders.jpg", no_borders)

# 7) Rotation/Deskewing
new = cv2.imread("data/page_01_rotated.JPG")

import numpy as np

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

fixed = deskew(new)
cv2.imwrite("temp/rotated_fixed.jpg", fixed)

#
#
# # 7) Add in missing borders
# color = [25, 40, 255]
# top, bottom, left, right = [150]*4
#
# image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
# cv2.imwrite("temp/image_with_border.jpg", image_with_border)

'''----------------------------------------------Pytesseract--------------------------------------------------------'''

# pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# no_noise = "temp/no_noise.jpg"
# img = Image.open(no_noise)
bw = "temp/bw_image.jpg"
img = Image.open(bw)
ocr_result = pytesseract.image_to_string(img)

'''-------------------------------------------Post-Processing----------------------------------------------------'''
# 1) Fix syntax
# dict = enchant.Dict('en_US')
#
# words = ocr_result.split()
# delimiters = [word[-1] for word in words[:-1]] + ['']
#
# corrected_words = []
#
# for i in range (len(words)):
#     if not dict.check(words[i].strip(",.?!")):
#         suggestion = dict.suggest(words[i])
#         if len(suggestion) > 0:
#             words[i] = suggestion[0]
#         corrected_words.append(words[i])
#     else:
#         corrected_words.append(words[i].strip(",.?!"))
#
# corrected_text = ""
# for i in range(len(corrected_words)):
#     corrected_text += corrected_words[i] + delimiters[i]
#
# ocr_result = ' '.join(words)
#
# 2) Delete word (Delimiter) created by noise
ocr_result = re.sub(r'\b[A-Za-z]{1}\b', '', ocr_result)
ocr_result = re.sub(r'[^\x00-\x7F]+', ' ', ocr_result)
ocr_result = re.sub(r'[^a-zA-Z()~,;.?"\s]', '', ocr_result)

 # 3) Fix random break line
ocr_result = ocr_result.replace('.\n', '. ')
ocr_result = ocr_result.replace('?\n', '? ')
ocr_result = ocr_result.replace('!\n', '! ')

ocr_result = "\n".join([ll.rstrip() for ll in ocr_result.splitlines() if ll.strip()])

print(ocr_result)


# '''pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# image_file = "data/index_02.JPG"
# img = Image.open(image_file)
# ocr_result = pytesseract.image_to_string(img)
# print(ocr_result)'''
#
# '''image = cv2.imread("data/index_02.JPG")
# base_image = image.copy()
# # make document easier for computer to read
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("temp/index_gray.png", gray)
# # help to identify the structure of the document
# blur = cv2.GaussianBlur(gray, (7,7), 0)
# cv2.imwrite("temp/index_blur.png", blur)
# #
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# cv2.imwrite("temp/index_thresh.png", thresh)
# #
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,13))
# cv2.imwrite("temp/index_kernel.png", kernel)
# #
# dilate = cv2.dilate(thresh, kernel, iterations=1)
# cv2.imwrite("temp/index_dilate.png", dilate)
# #
# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0]) #organize every thing in cnts from left to right
#
# results = []
#
# for c in cnts:
#     x, y, w, h = cv2.boundingRect(c)
#     if h > 200 and w > 20:
#         roi = image[y:y+h, x:x+h]
#         cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 2)
#         ocr_result = pytesseract.image_to_string(roi)
#         ocr_result = ocr_result.split("\n")
#         for item in ocr_result:
#             results.append(item)
#
# cv2.imwrite("temp/index_bbox.png", image)
#
# entities = []
#
# for item in results:
#     item = item.strip().replace("\n","")
#     item = item.split(" ")[0]
#     if len(item) > 2:
#         if item[0] == "A" and "-" not in item:
#             item = item.split(".")[0].replace(",","").replace(";","")
#             entities.append(item)
#
# entities = (set(entities))
# print(entities)'''
#
# '''image = cv2.imread("data/sample_mgh.JPG")
# base_image = image.copy()
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (7,7), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 50))
# dilate = cv2.dilate(thresh, kernel, iterations=1)
#
# cv2.imwrite("temp/sample_dilated.png", dilate)
#
# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
#
# for c in cnts:
#     x, y, w, h = cv2.boundingRect(c)
#     if h > 200 and w > 250:
#         roi = base_image[y:y+h, x:x+w]
#         cv2.rectangle(image, (x,y), (x+w,y+h), (36, 255, 12), 2)
#
# cv2.imwrite("temp/sample_box.png", image)
#
# ocr_result_new = pytesseract.image_to_string(roi)
# print(ocr_result_new)'''
#
# '''image = cv2.imread("data/sample_mgh.JPG")
# im_h, im_w, im_d = image.shape
# base_image = image.copy()
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (7,7), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
# dilate = cv2.dilate(thresh, kernel, iterations=1)
#
# cv2.imwrite("temp/sample_dilated2.png", dilate)
#
# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
#
# for c in cnts:
#     x, y, w, h = cv2.boundingRect(c)
#     if h < 20 and w > 250:
#         roi = base_image[0:y+h, 0:x+im_w]
#         cv2.rectangle(image, (x,y), (x+w,y+h), (36, 255, 12), 2)
#
# cv2.imwrite("temp/sample_box2.png", image)'''
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
