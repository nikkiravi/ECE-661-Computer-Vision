# Name: Nikita Ravi
# Class: ECE 66100
# Homework #10
# Deadline: 12/09/2022

# Import Modules
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from autoencoder import evaluate_autoencoder

def get_images(path):
	filenames = [image for image in os.listdir(path) if os.path.isfile(os.path.join(path, image))]
	images = [cv2.imread(os.path.join(path, image)) for image in os.listdir(path) if os.path.isfile(os.path.join(path, image))]

	return images, filenames

def display_image(image, points=False):
	def click_event(event, x, y, flags, params):
		# This function was inspired by https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
		if(event == cv2.EVENT_LBUTTONDOWN):
			print(x, y)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.circle(image, (x, y), 1, (0,0,255), thickness=-1)
			cv2.putText(image, str(x) + ',' +
						str(y), (x,y), font,
						1, (0, 0, 255), 2)
			cv2.imshow('window', image)

	cv2.imshow("window", image)
	if(points):
		cv2.setMouseCallback('window', click_event)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	quit()

def calculate_distance(trainFeatureVectors, testFeatueVector):
	dist = np.linalg.norm(trainFeatureVectors - testFeatueVector, axis=0)
	return dist

class KNearestNeighbors:
	def __init__(self, k):
		self.k = k
	
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		X_test = X_test.T
		predicted_classes = np.zeros(len(X_test))
		for idx, X in enumerate(X_test):
			dist = calculate_distance(self.X_train, X.reshape(-1, 1))
			sorted_dist_idx = np.argsort(dist)
			predicted_class = self.y_train[sorted_dist_idx[:self.k]]
			
			frequency = np.bincount(predicted_class)
			label = np.argmax(frequency)
			predicted_classes[idx] = label
			
		return predicted_classes

class PCA:
	def __init__(self, X_train, y_train, X_test, y_test, P, k):
		self.k = k
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.P = P

	def train(self):
		self.global_mean = np.mean(self.X_train, axis=1).reshape(-1, 1)
		X = self.X_train - self.global_mean

		_, _, u = np.linalg.svd(X.T @ X)
		w = X @ u # True eigenvectors of covariance
		w /= np.linalg.norm(w, axis=0)

		self.wp = w[:, :self.P] # Preserving P eigenvectors
		trainFeature = self.wp.T @ X # Project onto eigenspace

		self.knn = KNearestNeighbors(self.k)
		self.knn.fit(trainFeature, self.y_train)

	def test(self):
		X = self.X_test - self.global_mean
		testFeature = self.wp.T @ X # Project onto eigenspace
		predicted_classes = self.knn.predict(testFeature)

		acc = np.sum((predicted_classes - self.y_test) == 0) / np.float(self.y_test.size) * 100
		return acc

class LDA:
	def __init__(self, X_train, y_train, X_test, y_test, P, NUM_CLASSES, k):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.P = P
		self.C = NUM_CLASSES
		self.k = k

	def train(self):
		self.P = self.C - 1 if self.P > self.C - 1 else self.P
		self.global_mean = np.mean(self.X_train, axis=1)

		self.class_means = np.zeros((self.X_train.shape[0], self.C))
		for idx in range(self.C):
			self.class_means[:, idx] = np.mean(self.X_train[:, self.y_train == idx + 1], axis=1)
		
		"""
		It can be shown that when Sw is isotropic (all classes have identical variances in all of the same principal directions), the
		LDA eigenvectors are the eigenvectors of the Sb matrix. These correspond to the space spanned by the |C| - 1 mean difference
		m_i - m. 
		"""
		w = self.class_means - self.global_mean.reshape(-1, 1) # Equation 26
		self.wp = w[:, :self.P] # Preserve P eigenvectors

		trainFeature = self.wp.T @ (self.X_train - self.global_mean.reshape(-1, 1)) # Project onto eigenspace
		self.knn = KNearestNeighbors(self.k)
		self.knn.fit(trainFeature, self.y_train)

	def test(self):
		X = self.X_test - self.global_mean.reshape(-1, 1)
		testFeature = self.wp.T @ X

		predicted_classes = self.knn.predict(testFeature)
		acc = np.sum((predicted_classes - self.y_test) == 0) / np.float(self.y_test.size) * 100
		return acc

def vectorize_face_images(X):
	Xmean = np.mean(X)
	Xstd = np.std(X)
	X_normalized = (X - Xmean) / Xstd

	return X_normalized

def create_face_datasets(IMAGE_SIZE, N, images, filenames):
	X = np.zeros((IMAGE_SIZE, N), dtype=np.float32)
	y = np.zeros(N, dtype=np.int64)

	for idx, image in enumerate(images):
		filename = filenames[idx]
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

		X[:, idx] = gray.flatten()
		y[idx] = int(filename.split("_")[0])

	X = vectorize_face_images(X) # Normalize the images so the mean is at zero and standard deviation at 1
	return X, y

def plot_accuracies(PCA_LDA_accuracies, autoencoder_accuracies):
	pca, lda = PCA_LDA_accuracies

	plt.plot(range(1, 20 + 1), pca, label="PCA Accuracies")
	plt.plot(range(1, 20 + 1), lda, label="LDA Accuracies")
	plt.plot([3, 8, 16], autoencoder_accuracies, label="Autoencoder") # From autoencoder script results

	plt.title("Accuracies obtained for PCA, LDA, Autoencoder")
	plt.xlabel("P Eigenvectors")
	plt.ylabel("% Accuracy")

	plt.legend()
	plt.show()

def task1():
	# Constants
	TRAIN_DIR = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw10/FaceRecognition/train"
	TEST_DIR = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw10/FaceRecognition/test"
	N = 630 # Number of images in both training and testing dataset
	IMAGE_SIZE = 128 * 128 # 128 x 128 resolution 
	NUM_CLASSES = 30 # Number of classes
	MAX_P = 20 # Maximum number of eigenvalues we want
	NUM_NEIGHBORS = 1 # 1-NN Classification

	trainImages, trainFileNames = get_images(TRAIN_DIR)
	testImages, testFileNames = get_images(TEST_DIR)

	X_train, y_train = create_face_datasets(IMAGE_SIZE, N, trainImages, trainFileNames) # X_train shape: (128x128, 630), y_train shape: (630, 1)
	X_test, y_test = create_face_datasets(IMAGE_SIZE, N, testImages, testFileNames) # X_test shape: (128x128, 630), y_test shape: (630, 1)
	accuracies_pca_lda = []
	
	for mode in ["pca", "lda"]:
		print("============"+mode.upper()+"==================")
		accuracies_for_each_P = []
		for P in range(1, MAX_P+1):
			accuracy = None
			if(mode == "pca"):
				pca = PCA(X_train, y_train, X_test, y_test, P, k=NUM_NEIGHBORS)
				pca.train()
				accuracy = pca.test()

			elif(mode == "lda"):
				lda = LDA(X_train, y_train, X_test, y_test, P, NUM_CLASSES, k=NUM_NEIGHBORS)
				lda.train()
				accuracy = lda.test()

			accuracies_for_each_P.append(accuracy)
			print(f"Accuracy at {P} eigenvectors: {accuracy}")
		accuracies_pca_lda.append(accuracies_for_each_P)

	return accuracies_pca_lda

def task2():
	accuracies_for_p = []
	for p in [3, 8, 16]:
		X_train, y_train, X_test, y_test = evaluate_autoencoder(p, training=False)
		knn = KNearestNeighbors(k=1)
		knn.fit(X_train.T, y_train)
		predicted_classes = knn.predict(X_test.T)

		accuracy = np.sum((predicted_classes - y_test) == 0) / np.float(y_test.size) * 100
		print(f"Accuracy at {p} eigenvectors: {accuracy}")
		accuracies_for_p.append(accuracy)

	return accuracies_for_p
	
def create_feature_matrix(IMAGE_SIZE, height, width):
	# Haar Kernel Feature Extraction inspired by Fangda Li's code
	num_features = 47232 # Initially tried with 60000 but index ended with 47232
	feature_matrix = np.zeros((num_features, IMAGE_SIZE), dtype=int)
	index = 0 # Keep track of the index of feature_matric when appending to count number of features
	offset = 2 # offset of pixels from image borders
	
	# Horizontal haar filter
	width_step, height_step = 2, 1
	for i in range(1, height, height_step): # row multiplier
		for j in range(1, width, width_step): # height multiplier
			for y in range(offset, height - height_step * i + 1 - offset): # Go through the image
				for x in range(offset, width - width_step * j + 1 - offset):
					feature_matrix[index, y * height + x] = 1.0
					feature_matrix[index, y * height + x + width_step * j//2] = -2.0
					feature_matrix[index, y * height + x + width_step * j] = 1.0
					feature_matrix[index, (y + height_step * i) * height + x] = -1.0
					feature_matrix[index, (y + height_step * i) * height + x + width_step * j//2] = 2.0
					feature_matrix[index, (y + height_step * i) * height + x + width_step * j] = -1.0
					index += 1

	# Vertical haar filter
	width_step, height_step = 1, 2
	for i in range(1, height, height_step): # row multiplier
		for j in range(1, width, width_step): # height multiplier
			for y in range(offset, height - height_step * i + 1 - offset): # Go through the image
				for x in range(offset, width - width_step * j + 1 - offset):
					feature_matrix[index, y * height + x] = -1.0
					feature_matrix[index, y * height + x + width_step * j] = 1.0
					feature_matrix[index, (y + height_step * i//2) * height + x] = 2.0
					feature_matrix[index, (y + height_step * i//2) * height + x + width_step * j] = -2.0
					feature_matrix[index, (y + height_step * i) * height + x//2] = -1.0
					feature_matrix[index, (y + height_step * i) * height + x + width_step * j] = 1.0
					index += 1

	print(f"The number of features are: {index}")
	return feature_matrix

class CascadedAdaBoost:
	# This class was inspired by Fangda Li's code
	def __init__(self, NUM_CASCADES, NUM_WEAK_CLASSIFIERS, IMAGE_SIZE, height, width):
		self.NUM_CASCADES = NUM_CASCADES
		self.NUM_WEAK_CLASSIFIERS = NUM_WEAK_CLASSIFIERS
		self.IMAGE_SIZE = IMAGE_SIZE
		self.height = height
		self.width = width
		self.cascaded_adaboost = []
		self.feature_matrix = create_feature_matrix(self.IMAGE_SIZE, self.height, self.width)
		self.num_train_positives = None
		self.num_train_negatives = None
		self.X_train = None
		self.y_train = None

	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
		self.num_train = self.y_train.size

	def train(self):
		self.trainFeatureVectors = self.feature_matrix @ self.X_train
		self.num_train_positives = int(np.sum(self.y_train))
		self.num_train_negatives = len(self.y_train) - self.num_train_positives

		self.positive_train_feature_vector = self.trainFeatureVectors[:, self.y_train == 1]
		self.negative_train_feature_vector = self.trainFeatureVectors[:, self.y_train == 0]

		current_positives_feature_vectors = self.positive_train_feature_vector
		current_negatives_feature_vectors = self.negative_train_feature_vector

		false_positive_train = []
		false_negative_train = []
		true_positive_train = []
		accuracy_train = []

		false_positive_test = []
		false_negative_test = []
		true_positive_test = []
		accuracy_test = []

		for idx in range(self.NUM_CASCADES):
			print(f"Training AdaBoost with Cascade Number {idx+1}")
			current_adaboost_classifier = self.__add_new_adaboost_classifier()
			current_adaboost_classifier.set_training_feature_vectors(current_positives_feature_vectors, current_negatives_feature_vectors)

			for weak in range(self.NUM_WEAK_CLASSIFIERS):
				print(f"Adding weak classifier number: {weak + 1}")
				current_adaboost_classifier.add_weak_classifier()
			
			self.FalsePositiveIdx, FalsePositive, FalseNegative, TruePositive, Accuracy = self.__classify_training_data()
			false_positive_train.append(FalsePositive)
			false_negative_train.append(FalseNegative)
			true_positive_train.append(TruePositive)
			accuracy_train.append(Accuracy)

			current_negatives_feature_vectors = self.negative_train_feature_vector[:, self.FalsePositiveIdx - self.num_train_positives]
			FalsePositive, FalseNegative, TruePositive, Accuracy = self.__classify_testing_data()
			false_positive_test.append(FalsePositive)
			false_negative_test.append(FalseNegative)
			true_positive_test.append(TruePositive)
			accuracy_test.append(Accuracy)

		return false_positive_train, false_positive_test, false_negative_train, false_negative_test, true_positive_train, true_positive_test, accuracy_train, accuracy_test

	def __add_new_adaboost_classifier(self):
		adaboost = Adaboost()
		adaboost.set_feature_matrix(self.feature_matrix)
		self.cascaded_adaboost.append(adaboost)

		return adaboost

	def __classify_training_data(self):
		temp_feature_vector = self.trainFeatureVectors
		pos_idx = np.arange(self.num_train)

		for classifier in self.cascaded_adaboost:
			predicted = classifier.classify_feature_vectors(temp_feature_vector)
			temp_feature_vector = temp_feature_vector[:, predicted == 1]
			pos_idx = pos_idx[predicted == 1]

		# Sort TruePositives, FalsePositives, FalseNegatives
		FalsePositiveIdx = pos_idx[np.take(self.y_train, pos_idx) == 0]
		num_true_positives = np.sum(np.take(self.y_train, pos_idx))
		TruePositive = num_true_positives / self.num_train_positives
		FalsePositive = (pos_idx.size - num_true_positives) / self.num_train_negatives
		FalseNegative =  1 - TruePositive
		w = self.num_train_positives / (self.num_train_positives + self.num_train_negatives) 
		Accuracy = TruePositive * w + (1 - FalsePositive) * (1 - w)

		print("Training FP = %.4f, FN = %.4f, TP = %.4f, Acc = %.4f" % (FalsePositive, FalseNegative, TruePositive, Accuracy))
		return FalsePositiveIdx, FalsePositive, FalseNegative, TruePositive, Accuracy
	
	def set_testing(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test

	def __classify_testing_data(self):
		self.testFeatureVectors = self.feature_matrix @ self.X_test 
		temp_feature_vector = self.testFeatureVectors

		self.num_test_positives = int(np.sum(self.y_test))
		self.num_test_negatives = len(self.y_test) - self.num_test_positives

		num_test = self.y_test.size
		pos_idx = np.arange(num_test)

		for classifier in self.cascaded_adaboost:
			predicted = classifier.classify_feature_vectors(temp_feature_vector)
			temp_feature_vector = temp_feature_vector[:, predicted == 1]
			pos_idx = pos_idx[predicted == 1]

		# Sort TruePositives, FalsePositives, FalseNegatives
		FalsePositiveIdx = pos_idx[np.take(self.y_test, pos_idx) == 0]
		num_true_positives = np.sum(np.take(self.y_test, pos_idx))
		TruePositive = num_true_positives / self.num_test_positives
		FalsePositive = (pos_idx.size - num_true_positives) / self.num_test_negatives
		FalseNegative = 1 - TruePositive
		w = self.num_test_positives / (self.num_test_positives + self.num_test_negatives) 
		Accuracy = TruePositive * w + (1 - FalsePositive) * (1 - w)

		print("Testing FP = %.4f, FN = %.4f, TP = %.4f, Acc = %.4f" % (FalsePositive, FalseNegative, TruePositive, Accuracy))
		return FalsePositive, FalseNegative, TruePositive, Accuracy

class Adaboost:
	# This class was inspired by Fangda Li's code
	def __init__(self):
		self.weak_classifier_index = np.array([], dtype=int)
		self.weak_classifier_polarities = np.array([])
		self.weak_classifier_threshold = np.array([])
		self.weak_classifier_weights = np.array([])
		self.weak_classifier_results = np.array([])
		self.weak_classifier_weighted_results = None
		self.threshold = 1.0
		self.y_train = None
		self.train_sorted_idx = None
		self.train_feature_vector = None
		self.num_positive = None
		self.num_negative = None
		self.weights_for_sample = None

	def set_feature_matrix(self, feature_matrix):
		self.feature_matrix = feature_matrix

	def set_training_feature_vectors(self, positive_feature_vector, negative_feature_vector):
		self.NUM_FEATURES = positive_feature_vector.shape[0]
		self.num_positive = positive_feature_vector.shape[1] # shape: num_featurs=47232 x num_images
		self.num_negative = negative_feature_vector.shape[1]
		self.train_feature_vector = np.hstack((positive_feature_vector, negative_feature_vector))
		self.y_train = np.hstack((np.ones(self.num_positive), np.zeros(self.num_negative)))
		self.train_sorted_idx = np.argsort(self.train_feature_vector, axis=1)

		print(f"Number of positive training data: {self.num_positive} / negative training data: {self.num_negative}")
	
	def add_weak_classifier(self):
		# Initialize the weights for the first weak classifier
		if(not self.weak_classifier_index.size):
			self.weights_for_sample = np.zeros(self.y_train.size, dtype=float)
			self.weights_for_sample[self.y_train == 1] = 1 / (2 * self.num_positive)
			self.weights_for_sample[self.y_train == 0] = 1 / (2 * self.num_negative)

		# Normalize the weights
		else:
			self.weights_for_sample /= np.sum(self.weights_for_sample)

		# Get the best weak classifier that minimizes the error with the current weights
		best_feature_index, best_feature_polarity, best_feature_threshold, best_feature_error, best_feature_results = self.__get_best_weak_classifier()

		self.weak_classifier_index = np.append(self.weak_classifier_index, best_feature_index)
		self.weak_classifier_polarities = np.append(self.weak_classifier_polarities, best_feature_polarity)
		self.weak_classifier_threshold = np.append(self.weak_classifier_threshold, best_feature_threshold)

		# Confidence
		beta = best_feature_error / (1 - best_feature_error)

		# Trust Factor
		alpha = np.log(1 / np.abs(beta))
		self.weak_classifier_weights = np.append(self.weak_classifier_weights, alpha)
		e = np.abs(best_feature_results - self.y_train)

		# Update the weights
		self.weights_for_sample = self.weights_for_sample * beta ** (1 - e)

		# Adjust the threshold
		if(not len(self.weak_classifier_results)):
			self.weak_classifier_results = best_feature_results.reshape(-1,1)
		else: 
			self.weak_classifier_results = np.hstack((self.weak_classifier_results, best_feature_results.reshape(-1,1)))

		self.weak_classifier_weighted_results = np.dot(self.weak_classifier_results, self.weak_classifier_weights)
		self.threshold = min(self.weak_classifier_weighted_results[self.y_train == 1])

	def __get_best_weak_classifier(self):
		feature_errors = np.zeros(self.NUM_FEATURES)
		feature_threshold = np.zeros(self.NUM_FEATURES)
		feature_polarity = np.zeros(self.NUM_FEATURES)
		feature_sorted_idx = np.zeros(self.NUM_FEATURES, dtype=int)
		
		Tpos = np.sum(self.weights_for_sample[self.y_train == 1])
		Tneg = np.sum(self.weights_for_sample[self.y_train == 0])
		for f in range(self.NUM_FEATURES):
			sorted_weights = self.weights_for_sample[self.train_sorted_idx[f, :]]
			sorted_labels = self.y_train[self.train_sorted_idx[f, :]]

			Spos = np.cumsum(sorted_labels * sorted_weights)
			Sneg = np.cumsum(sorted_weights) - Spos

			Epos = Spos + Tneg - Sneg
			Eneg = Sneg + Tpos - Spos

			polarities = np.zeros(self.num_positive + self.num_negative)
			polarities[Epos > Eneg] = -1
			polarities[Epos <= Eneg] = 1
			# print(Epos > Eneg)

			errors = np.minimum(Epos, Eneg)
			sorted_idx = np.argmin(errors)

			min_error_sample_idx = self.train_sorted_idx[f, sorted_idx]
			min_error = np.min(errors)

			threshold = self.train_feature_vector[f, min_error_sample_idx]
			polarities = polarities[sorted_idx]

			feature_errors[f] = min_error
			feature_threshold[f] = threshold
			feature_polarity[f] = polarities
			feature_sorted_idx[f] = sorted_idx

		best_feature_index = np.argmin(feature_errors)
		best_feature_threshold = feature_threshold[best_feature_index]
		best_feature_error = feature_errors[best_feature_index]
		best_feature_polarity = feature_polarity[best_feature_index]

		best_feature_results = np.zeros(self.num_positive + self.num_negative)
		best_sorted_index = feature_sorted_idx[best_feature_index]
		if(best_feature_polarity == 1):
			best_feature_results[self.train_sorted_idx[best_feature_index, best_sorted_index:]] = 1
		else:
			best_feature_results[self.train_sorted_idx[best_feature_index, :best_sorted_index]] = 1
		
		return best_feature_index, best_feature_polarity, best_feature_threshold, best_feature_error, best_feature_results

	def classify_feature_vectors(self, train_feature_vector):
		weak_classifiers = train_feature_vector[self.weak_classifier_index, :]

		polarity_vector = self.weak_classifier_polarities.reshape(-1, 1)
		threshold_vector = self.weak_classifier_threshold.reshape(-1, 1)

		# Predictions made by weak classifier
		weak_classifier_predictions = weak_classifiers * polarity_vector > threshold_vector * polarity_vector
		weak_classifier_predictions[weak_classifier_predictions == True] = 1
		weak_classifier_predictions[weak_classifier_predictions == False] = 0

		# Apply weak classifier weights
		strong_classifier_result = self.weak_classifier_weights @ weak_classifier_predictions

		# Apply strong classifier threshold
		final_predictions = np.zeros(strong_classifier_result.size)
		final_predictions[strong_classifier_result >= self.threshold] = 1
		return final_predictions

def get_integral_image(image):
	# Return integral representation of the image
	# https://towardsdatascience.com/understanding-face-detection-with-the-viola-jones-object-detection-framework-c55cc2a9da14
	# Cumulative sum of above and to the left of the current pixel
	return np.cumsum(np.cumsum(image, axis=0), axis=1)

def create_car_datasets(IMAGE_SIZE, positive_images, negative_images):
	N = len(positive_images) + len(negative_images)
	positive_flattened_data = np.zeros((IMAGE_SIZE, len(positive_images)))
	negative_flattened_data = np.zeros((IMAGE_SIZE, len(negative_images)))

	for idx, image in enumerate(positive_images):
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY) if len(image.shape) == 3 else image
		image = get_integral_image(image)
		positive_flattened_data[:, idx] = image.flatten()

	positive_labels = np.ones(len(positive_images))

	for idx, image in enumerate(negative_images):
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY) if len(image.shape) == 3 else image
		image = get_integral_image(image)
		negative_flattened_data[:, idx] = image.flatten()

	negative_labels = np.zeros(len(negative_images))

	X = np.hstack((positive_flattened_data, negative_flattened_data))
	y = np.hstack((positive_labels, negative_labels))
	return X, y

def plot_task3_results(train, test, title):
	plt.plot(range(1, len(train) + 1), train, label="train")
	plt.plot(range(1, len(test) + 1), test, label="test")
	
	plt.title(title)
	plt.xlabel("Cascaded Layer")
	plt.ylabel(title)

	plt.legend()
	plt.show()

def task3():
	# Constants
	TRAIN_DIR = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw10/CarDetection/train"
	TEST_DIR = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw10/CarDetection/test"
	IMAGE_SIZE = 40 * 20 # 40x20 image
	HEIGHT, WIDTH = 20, 40
	NUM_CASCADES = 10
	NUM_WEAK_CLASSIFIERS = 20

	train_positive_images, train_positive_filenames = get_images(os.path.join(TRAIN_DIR, "positive")) 
	train_negative_images, train_negative_filenames = get_images(os.path.join(TRAIN_DIR, "negative")) 
	test_positive_images, test_positive_filenames = get_images(os.path.join(TEST_DIR, "positive")) 
	test_negative_images, test_negative_filenames = get_images(os.path.join(TEST_DIR, "negative")) 

	X_train, y_train = create_car_datasets(IMAGE_SIZE, train_positive_images, train_negative_images) # X_train: (800, 2468), y_train: (2468,)
	X_test, y_test = create_car_datasets(IMAGE_SIZE, test_positive_images, test_negative_images) # X_test: (800, 618), y_test: (618,)

	classifier = CascadedAdaBoost(NUM_CASCADES, NUM_WEAK_CLASSIFIERS, IMAGE_SIZE, HEIGHT, WIDTH)
	classifier.fit(X_train, y_train)
	classifier.set_testing(X_test, y_test)
	false_positive_train, false_positive_test, false_negative_train, false_negative_test, true_positive_train, true_positive_test, accuracy_train, accuracy_test = classifier.train()

	plot_task3_results(false_positive_train, false_positive_test, "False Positive Rate")
	plot_task3_results(false_negative_train, false_negative_test, "False Negative Rate")
	plot_task3_results(true_positive_train, true_positive_test, "True Positive Rate")
	plot_task3_results(accuracy_train, accuracy_test, "Accuracy Rate")
	
if __name__ == "__main__":
	print("=====TASK 1=====")
	pca_lda_accuracies = task1()

	print("=====TASK 2=====")
	autoencoder_accuracies = task2()
	plot_accuracies(pca_lda_accuracies, autoencoder_accuracies)

	print("=====TASK 3=====")
	task3()