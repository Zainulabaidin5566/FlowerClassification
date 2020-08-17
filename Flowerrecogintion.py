# Importing libs
from skimage.segmentation import chan_vese
import cv2
import os
import numpy as np
import math
from skimage.feature import greycomatrix, greycoprops
from scipy.spatial import distance
from scipy.stats import skew

# Creating empty list of training images
training_data = []


# Getting Training data-set
def image_resize():
    path_to_folder = r'C:\Users\Danyal\Desktop\6th Semester\Digitial Image Processing\DIP Semester Project\Flowers dataset\flowers_data'
    categories = ["1", "2", "3", "4"]
    # Reading all classes of images
    for category in categories:
        path = os.path.join(path_to_folder, category)
        for img in os.listdir(path):
            first_img = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            # Equalizing dimensions of all images
            first_img = cv2.resize(first_img, (300, 300))
            training_data.append(first_img)
    return training_data


# Empty List for test data
t_data1 = []


# Reading test images
def test_image_resize():
    path_to_folder = r'C:\Users\Danyal\Desktop\6th Semester\Digitial Image Processing\DIP Semester Project\Flowers dataset\test_images'
    for img in os.listdir(path_to_folder):
        first_img = cv2.imread(os.path.join(path_to_folder, img), cv2.IMREAD_COLOR)
        first_img = cv2.resize(first_img, (300, 300))
        t_data1.append(first_img)
    return t_data1


# Function for calculating Hu-moments
def HU_Moments(segmented):
    moments = cv2.moments(segmented)
    huMoments = cv2.HuMoments(moments)
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
    return huMoments


# Empty lists for GLCM outputs
contrast = []
dissimiarity = []
homogenity = []
energy = []
correlation = []


# Function for finding GLCM of images
def GLCM(first_img):
    result = greycomatrix(first_img, [1], [0], levels=256, symmetric=False, normed=False)
    my_GLCM = result[:, :, 0, 0]
    # Getting contrast, dissimilarity, homogeneity, energy and correlation
    c = greycoprops(result, prop='contrast')
    d = greycoprops(result, prop='dissimilarity')
    h = greycoprops(result, prop='homogeneity')
    e = greycoprops(result, prop='energy')
    cr = greycoprops(result, prop='correlation')
    return c, d, h, e, cr


# Assigning labels to different classes of images
def Labels():


    label = []
    for x in range(120):
        if x < 30:
            # Label for type 1 flower images i.e daffodils
            label.append(0)
        elif x >= 30 and x < 60:
            # Label for type 2 flower images  i.e lotus
            label.append(1)
        elif x >= 60 and x < 90:
            # Label for type 3 flower images i.e daisy
            label.append(2)
        else:
            # Label for type 4 flower images i.e bluebell
            label.append(3)

    return label


# Function for getting color moments of images
def color_moments(image):
    # Converting BGR images to HSV
    hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Getting Hue, saturation and value images
    H, S, V = cv2.split(hsvimage)
    # Finding mean, variance and skewness
    Hmean = np.mean(H)
    Hvar = np.var(H)
    Hskewness = skew(H, None)
    Smean = np.mean(S)
    Svar = np.var(S)
    Sskewness = skew(S, None)
    Vmean = np.mean(V)
    Vvar = np.var(V)
    Vskewness = skew(V, None)
    return Hmean, Hvar, Hskewness, Smean, Svar, Sskewness, Vmean, Vvar, Vskewness


# Creating empty lists for storing different features of images
segmented = []
Fmom = []
Smom = []
Tmom = []
Fourmom = []
Fifmom = []
Sixmom = []
Sevmom = []
Local_binary = []
Hist = []
Final_list = []
# calling function which reads training images
training_data = image_resize()
ch = 0
for i in training_data:
    # Splitting image to B, G and R planes
    image, G, R = cv2.split(i)
    # Applying chan-vese segmentation on blue plane
    cv = chan_vese(image, mu=0.5, lambda1=1, lambda2=1, tol=1e-3, max_iter=300,
                   dt=0.5, init_level_set="checkerboard", extended_output=True)
    ch = ch + 1
    print(ch)

    segmented.append(cv[0])

# New B, G and R planes for image
B = np.zeros([300, 300])
G = np.zeros([300, 300])
R = np.zeros([300, 300])

count = 0
suck = 0
for i in segmented:
    suck = suck + 1
    print(suck)
    # Getting shape for loop
    row = i.shape[0]
    col = i.shape[1]
    img = training_data[count]
    b, g, r = cv2.split(img)
    for x in range(row):
        for y in range(col):
            if i[x][y] == 1:
                # Saving only foreground pixels
                B[x][y] = b[x][y]
                G[x][y] = g[x][y]
                R[x][y] = r[x][y]
            else:
                B[x][y] = 0
                G[x][y] = 0
                R[x][y] = 0
    count = count + 1
    # Merging B, G and R planes for single colored image
    final = cv2.merge((B, G, R))
    final = final.astype(np.uint8)
    # Converting color image to gray
    grayImage = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    rr = grayImage.shape[0]
    cc = grayImage.shape[1]
    # Thresh holding image to get binary image
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    # Calling function to get Hu-moments of binary image
    huMoments = HU_Moments(blackAndWhiteImage)
    # Storing Hu-moments values in respective list
    for b in range(0, 7):
        if b == 0:
            Fmom.append(huMoments[b])
        elif b == 1:
            Smom.append(huMoments[b])
        elif b == 2:
            Tmom.append(huMoments[b])
        else:
            Fourmom.append(huMoments[b])

    # Getting contrast, dissimilarity etc. from GLCm function
    c, d, h, e, cr = GLCM(grayImage)
    # calling color moments function defined above
    m1, v1, s1, m2, v2, s2, m3, v3, s3 = color_moments(final)
    # Storing all the features in single list
    Final_list.append(
        (huMoments[0], huMoments[1], huMoments[2], huMoments[3], c, d, h, e, cr, m1, v1, s1, m2, v2, s2, m3, v3, s3))
# Getting labels by calling Labels function
label = Labels()

# Chan-vesed List for test images
chanvesed = []
s = 0
test = []
histo = [0] * 256
hj = 0
# getting test images by calling function defined above
t_data1 = test_image_resize()
print("length", t_data1)
for im in t_data1:
    showim = im
    print("Testing started")
    hj = hj + 1
    print(hj)
    # Applying chan-vese segmentation on blue plane and adding the resultant image in the list
    B, G, R = cv2.split(im)
    cv = chan_vese(B, mu=0.5, lambda1=1, lambda2=1, tol=1e-3, max_iter=300,
                   dt=0.5, init_level_set="checkerboard", extended_output=True)
    chanvesed.append(cv[0])
cdd = 0
# Getting foreground flower pixels only for test images
for j in chanvesed:
    cdd = cdd + 1
    print(cdd)
    # Empty B, G and r planes for storing flower pixel values
    BB = np.zeros([300, 300])
    GG = np.zeros([300, 300])
    RR = np.zeros([300, 300])
    row = j.shape[0]
    col = j.shape[1]
    test_img = t_data1[s]
    s = s + 1
    b, g, r = cv2.split(test_img)
    for x in range(row):
        for y in range(col):
            # Storing only flower pixels and every other pixel is set to 0
            if j[x][y] == 1:
                BB[x][y] = b[x][y]
                GG[x][y] = g[x][y]
                RR[x][y] = r[x][y]
            else:
                BB[x][y] = 0
                GG[x][y] = 0
                RR[x][y] = 0

    # Merging B, G and R planes to get single colored image
    final = cv2.merge((BB, GG, RR))
    final = final.astype(np.uint8)
    # Getting gray image
    grayImage = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    # Converting gray image to black and white image
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    # using black and white image for finding Hu-moments
    huMoments = HU_Moments(blackAndWhiteImage)
    # Getting color moments. Functions are defined above
    m1, v1, s1, m2, v2, s2, m3, v3, s3 = color_moments(final)
    # getting contrast, homogeneity etc. from GLCM function
    c, d, h, e, cr = GLCM(grayImage)
    # Storing test image features
    test.append(
        (huMoments[0], huMoments[1], huMoments[2], huMoments[3], c, d, h, e, cr, m1, v1, s1, m2, v2, s2, m3, v3, s3))

# distance list
d = []

# Flowers
Daffodil = 0
Lotus = 0
Daisy = 0
BlueBell = 0
# Predictions list
predicted_list = []

c = test[0]
l_count = 0
# Finding euclidean distance and storing in distance list
for y in Final_list:
    dist = distance.euclidean(c, y)
    d.append((dist, label[l_count]))
    l_count = l_count + 1
# Sorting distance list
d.sort()
print(d)
list1 = []
# Using k = 5
k = 5
for i in range(k):
    list1.append(d[i])
for p in list1:
    # Checking flowers using their labels
    if p[1] == 0:
        Daffodil = Daffodil + 1
    elif p[1] == 1:
        Lotus = Lotus + 1
    elif p[1] == 2:
        Daisy = Daisy + 1
    elif p[1] == 3:
        BlueBell = BlueBell + 1

# Predicting results
if Daffodil > Lotus and Daffodil > Daisy and Daffodil > BlueBell:
    predicted_list.append('Daffodil')
elif Lotus > Daffodil and Lotus > Daisy and Lotus > BlueBell:
    predicted_list.append('Lotus')
elif Daisy > Daffodil and Daisy > Lotus and Daisy > BlueBell:
    predicted_list.append('Daisy')
elif BlueBell > Daffodil and BlueBell > Lotus and BlueBell > Daisy:
    predicted_list.append('BlueBell')
else:
    # Using k = 4 and re running same steps
    k = 4
    for i in range(k):
        list1.append(d[i])
    for p in list1:
        if p[1] == 0:
            Daffodil = Daffodil + 1
        elif p[1] == 1:
            Lotus = Lotus + 1
        elif p[1] == 2:
            Daisy = Daisy + 1
        elif p[1] == 3:
            BlueBell = BlueBell + 1
    # Predictions
    if Daffodil > Lotus and Daffodil > Daisy and Daffodil > BlueBell:
        predicted_list.append('Daffodil')
    elif Lotus > Daffodil and Lotus > Daisy and Lotus > BlueBell:
        predicted_list.append('Lotus')
    elif Daisy > Daffodil and Daisy > Lotus and Daisy > BlueBell:
        predicted_list.append('Daisy')
    elif BlueBell > Daffodil and BlueBell > Lotus and BlueBell > Daisy:
        predicted_list.append('BlueBell')
    else:
        # Using k = 3
        k = 3
        for i in range(k):
            list1.append(d[i])
        for p in list1:
            if p[1] == 0:
                Daffodil = Daffodil + 1
            elif p[1] == 1:
                Lotus = Lotus + 1
            elif p[1] == 2:
                Daisy = Daisy + 1
            elif p[1] == 3:
                BlueBell = BlueBell + 1
        # Predictions
        if Daffodil > Lotus and Daffodil > Daisy and Daffodil > BlueBell:
            predicted_list.append('Daffodil')
        elif Lotus > Daffodil and Lotus > Daisy and Lotus > BlueBell:
            predicted_list.append('Lotus')
        elif Daisy > Daffodil and Daisy > Lotus and Daisy > BlueBell:
            predicted_list.append('Daisy')
        elif BlueBell > Daffodil and BlueBell > Lotus and BlueBell > Daisy:
            predicted_list.append('BlueBell')
print('The Flower is a ', predicted_list)
cv2.putText(showim, predicted_list[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
      1.0, (0, 0, 255), 3)
cv2.imshow("Image", showim)
cv2.waitKey(0)