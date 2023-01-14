# Name: Nikita Ravi
# Class: ECE 66100
# Homework #7
# Deadline: 11/02/2022

# Import Modules
import cv2
import math
import numpy as np
import os
import re
import BitVector as bv
from pprint import pprint
from vgg import VGG19
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import seaborn as sn


def get_images(path):
    images = {"cloudy": [], "rain": [], "shine": [], "sunrise": []}
    names = {"cloudy": [], "rain": [], "shine": [], "sunrise": []}

    pattern = re.compile(r"([A-Za-z]+)([0-9]+)")
    for idx, image_name in enumerate(sorted(os.listdir(path)[::-1])):
        if(image_name != '.DS_Store' and image_name != "rain141.jpg" and image_name != "shine131.jpg"):
            group = pattern.findall(image_name)[0][0]
            names[group].append(image_name)

            img = cv2.imread(os.path.join(path, image_name))
            images[group].append(img)

    return images, names

def display_image(image):
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    quit()

def get_neighboring_pixels(R=1, P=8):
    neighbors = np.zeros((P, 2))
    for p in range(P):
        neighbors[p][0] = R * math.sin(2 * math.pi * p / P)
        neighbors[p][1] = R * math.cos(2 * math.pi * p / P)

    return neighbors

def get_image_pixel_value_at_p(image, k_base, l_base, delta_k, delta_l):
    # This function was inspired by Professor Kak's Local Binary Pattern Code from the Lecture Notes
    if(delta_k < 0.001 and delta_l < 0.001):
        image_pixel_val_at_p = float(image[k_base][l_base])

    elif(delta_l < 0.001):
        image_pixel_val_at_p = (1 - delta_k) * image[k_base][l_base] + delta_k * image[k_base + 1][l_base]
    
    elif(delta_k < 0.001):
        image_pixel_val_at_p = (1 - delta_l) * image[k_base][l_base] + delta_k * image[k_base][l_base + 1]
    
    else:
        image_pixel_val_at_p = (1 - delta_k) * (1 - delta_l) * image[k_base][l_base] + \
                (1 - delta_k) * delta_l * image[k_base][l_base + 1] + \
                delta_k * delta_l * image[k_base + 1][l_base + 1] + \
                delta_k * (1 - delta_l) * image[k_base + 1][l_base]

    return image_pixel_val_at_p

def get_encoding(bv_runs, lbp_hist):
    # This function was inspired by Professor Kak's Local Binary Pattern Code from the Lecture Notes
    if(len(bv_runs) > 2):
        lbp_hist[P + 1] += 1

    elif(len(bv_runs) == 1 and bv_runs[0][0] == '1'):
        lbp_hist[P] += 1
    
    elif(len(bv_runs) == 1 and bv_runs[0][0] == '0'):
        lbp_hist[0] += 1
    
    else:
        lbp_hist[len(bv_runs[1])] += 1

    return lbp_hist

def obtain_minimum_bitvec(pattern):
    # Calculate the minimum binary vector for the pattern obtained to get maximal amount of information for discriminating texture features in the image
    # This function was inspired by Professor Kak's Local Binary Pattern Code from the Lecture Notes
    bitvec = bv.BitVector(bitlist = pattern)
    intervals_for_circular_shifts = [int(bitvec << 1) for _ in range(P)]
    min_bitvec = bv.BitVector(intVal = min(intervals_for_circular_shifts), size=P)

    return min_bitvec

def generate_feature_vector_lbp(image, R=1, P=8):
    # This function was inspired by Professor Kak's Local Binary Pattern Code from the Lecture Notes
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape 
    total_num_pixels = height * width

    neighbors = get_neighboring_pixels(R, P) # Get all the pixels around the pixel at the origin at a specific radius R
    lbp_hist = {t:0 for t in range(P + 2)} # LBP Histogram

    row_max, col_max = width - R, height - R
    for i in range(R, row_max):
        for j in range(R, col_max):
            pattern = []
            for p in range(P):
                del_l, del_k = neighbors[p][0], neighbors[p][1]
                del_k = 0 if abs(del_k) < 0.001 else del_k
                del_l = 0 if abs(del_l) < 0.001 else del_l

                k, l = i + del_k, j + del_l
                k_base, l_base = int(k), int(l) # Corner coordinates for A, B, C, D
                delta_k, delta_l = k - k_base, l - l_base

                image_pixel_val_at_p = get_image_pixel_value_at_p(image, k_base, l_base, delta_k, delta_l)
                
                if(image_pixel_val_at_p >= image[i][j]):
                    pattern.append(1)
                else:
                    pattern.append(0)
            
            min_bitvec = obtain_minimum_bitvec(pattern)
            bv_runs = min_bitvec.runs() # number of groups of runs
            
            lbp_hist = get_encoding(bv_runs, lbp_hist)

    probability_hist = {key: value / total_num_pixels for key, value in lbp_hist.items()}
    return probability_hist

def generate_feature_vector_gram(image):
    feature_map_matrix = []

    # Initialize the vgg model
    vgg = VGG19()
    vgg.load_weights(r"hw07/HW7-Auxilliary/vgg_normalized.pth")

    feature_maps = vgg(image) # The feature maps contains layers each containing different pieces of information on style (edges, lines, dots, curves, etc.)
    for idx, style_map_matrix in enumerate(feature_maps):
        feature_map_matrix.append(style_map_matrix.flatten())

    feature_map_matrix = np.array(feature_map_matrix)
    gram_matrix = np.matmul(feature_map_matrix, feature_map_matrix.T) # 512x512 matrix

    # Randomly select 1024 samples using the upper triangle
    upper_triangle_gram_indices = np.triu_indices(gram_matrix.shape[0])
    upper_triangle_gram = gram_matrix[upper_triangle_gram_indices]
    
    mid_idx = len(upper_triangle_gram) // 2
    feature_vector = upper_triangle_gram[mid_idx:(mid_idx + 1024)]

    return gram_matrix, feature_vector

def plot_histogram(feature_vector, extraction, image_name):
    plt.figure()
    colors = plt.cm.get_cmap('tab20c')
    random_color_idx = np.random.rand()
    plt.bar(range(len(feature_vector)), feature_vector, width=0.8, color=colors(random_color_idx), edgecolor = "black")
    plt.title(extraction + "Histogram for " + image_name)
    plt.savefig("hw07/histogram_plots/" + extraction + "_" + image_name)

def plot_gram_matrix(gram_matrix, image_name):
    gram_matrix += 0.001
    mapping, log_norm = plt.cm.gray, colors.LogNorm()
    gram_plot = mapping(log_norm(gram_matrix))
    plt.imsave(fname=r"hw07/gram_matrix_plots/" + image_name, arr=gram_plot, format="jpg")

def generate_feature_matrix(image_dict, image_names, extraction, path, R=1, P=8, train=True):
    feature_matrix, group_matrix = [], []
    for key, images in image_dict.items():
        for idx, image in enumerate(images):
            image_name = image_names[key][idx]
            if(extraction == "lbp"):
                image_resized = cv2.resize(image, dsize=(64, 64))
                feature_vector = [val for key, val in generate_feature_vector_lbp(image_resized, R, P).items()]

                if(idx == 1 and train):
                    plot_histogram(feature_vector, extraction, image_name)

            elif(extraction == "gram"):
                image_resized = cv2.resize(image, dsize=(256, 256))
                gram_matrix, feature_vector = generate_feature_vector_gram(image_resized)

                if(idx == 1 and train):
                    plot_gram_matrix(gram_matrix, image_name)

            feature_matrix.append(feature_vector)
            group_matrix.append(key)

    np.savez_compressed(path, feature_matrix=feature_matrix, groups=group_matrix)

def load_saved_matrix(path):
    loaded = np.load(path)
    feature_matrix, group_matrix = np.matrix(loaded["feature_matrix"], dtype=np.float32), np.array(loaded["groups"])

    return feature_matrix, group_matrix

def nominal_encoding(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le

def train_support_vector_machine(feature_matrix, group_encoded, kernel="rbf"):
    clf = svm.SVC(kernel=kernel)
    clf.fit(feature_matrix, group_encoded)
    return clf

def predict_group_name(clf, feature_matrix):
    yHat = clf.predict(feature_matrix)
    return yHat

def plot_confusion_matrix(cnf_matrix, classes, title="LBP"):
    ax = sn.heatmap(cnf_matrix, annot=True, cmap='Blues')

    ax.set_title("Confusion Matrix for " + title +"\n\n");
    ax.set_xlabel('\nActual Values')
    ax.set_ylabel('Predicted Values ')

    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    plt.show()
    plt.savefig("hw07/confusion_matrix/" + title + ".jpg")

def create_confusion_matrix(y_test, yHat, le, title):
    y_test_decoded, yHat_decoded = list(le.inverse_transform(y_test)), list(le.inverse_transform(yHat))
    classes = list(set(y_test_decoded + yHat_decoded))

    cnf_matrix = confusion_matrix(y_test_decoded, yHat_decoded, labels=classes).T
    plot_confusion_matrix(cnf_matrix, classes, title=title)

def create_classification_report(y_test, yHat, le):
    y_test_decoded, yHat_decoded = list(le.inverse_transform(y_test)), list(le.inverse_transform(yHat))
    classes = list(set(y_test_decoded + yHat_decoded))

    report = classification_report(y_test_decoded, yHat_decoded, labels=classes)
    print(report)


if __name__ == "__main__":
    train_path = r"hw07/HW7-Auxilliary/data/training"
    test_path = r"hw07/HW7-Auxilliary/data/testing"

    train_images, train_names = get_images(train_path)
    test_images, test_names = get_images(test_path)
    
    mode = "Classify"

    if(mode == "Extract"):
        ### Task 1 - Local Binary Pattern
        P = 8 # Number of neighboring pixels
        R = 1 # Radius of circle neighborhood

        generate_feature_matrix(train_images, train_names, "lbp", r"hw07/lbp_matrix/trained_feature_matrix", R=1, P=8)
        generate_feature_matrix(test_images, test_names, "lbp", r"hw07/lbp_matrix/testing_feature_matrix", R=1, P=8, train=False)

        ### Task 2 - Gram Matrix Generation
        generate_feature_matrix(train_images, train_names, "gram", r"hw07/gram_matrix/trained_feature_matrix")
        generate_feature_matrix(test_images, test_names, "gram", r"hw07/gram_matrix/testing_feature_matrix", train=False)

    elif(mode == "Classify"):
        ### Task 3 - Building an Image Classifier Pipeline
        train_lbp_feature_matrix, train_lbp_group_matrix = load_saved_matrix(r"hw07/lbp_matrix/trained_feature_matrix.npz")
        test_lbp_feature_matrix, test_lbp_group_matrix = load_saved_matrix(r"hw07/lbp_matrix/testing_feature_matrix.npz")
        train_gram_feature_matrix, train_gram_group_matrix = load_saved_matrix(r"hw07/gram_matrix/trained_feature_matrix.npz")
        test_gram_feature_matrix, test_gram_group_matrix = load_saved_matrix(r"hw07/gram_matrix/testing_feature_matrix.npz")

        #=======================================================TRAIN===========================================================================
        label_encoder_lbp = nominal_encoding(train_lbp_group_matrix)
        label_encoder_gram = nominal_encoding(train_gram_group_matrix)

        train_lbp_group_encoded = label_encoder_lbp.transform(train_lbp_group_matrix)
        train_gram_group_encoded = label_encoder_gram.transform(train_gram_group_matrix)

        svm_lbp = train_support_vector_machine(train_lbp_feature_matrix, train_lbp_group_encoded)
        svm_gram = train_support_vector_machine(train_gram_feature_matrix, train_gram_group_encoded, kernel="linear")

        #=======================================================PREDICT=========================================================================
        yHat_lbp = predict_group_name(svm_lbp, test_lbp_feature_matrix)
        yHat_gram = predict_group_name(svm_gram, test_gram_feature_matrix)

        create_confusion_matrix(label_encoder_lbp.transform(test_lbp_group_matrix), np.array(yHat_lbp), label_encoder_lbp, title="LBP")
        print("Classification for LBP")
        create_classification_report(label_encoder_lbp.transform(test_lbp_group_matrix), np.array(yHat_lbp), label_encoder_lbp)

        create_confusion_matrix(label_encoder_gram.transform(test_gram_group_matrix), np.array(yHat_gram), label_encoder_gram, title="Gram")
        print("Classification for Gram")
        create_classification_report(label_encoder_gram.transform(test_gram_group_matrix), np.array(yHat_gram), label_encoder_gram)