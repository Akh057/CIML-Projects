####################
## Assignment 1 - Prototype based classification using convex combination of means
## @Akhil Polamarasetty
####################
import numpy as np
from numpy import linalg as LA

if __name__ == "__main__":
#load the training examples and test data
	X_seen = np.load('X_seen.npy',encoding = 'latin1')
	num_seen_class = np.shape(X_seen)[0]
	num_features = np.shape(X_seen[0])[1]
	Xtest = np.load('Xtest.npy',encoding = 'latin1')
	num_test_examples = np.shape(Xtest)[0]
	Ytest = np.load('Ytest.npy',encoding = 'latin1')
	# print("The parameters : num_seen_classes="+str(num_seen_class)+",num_features="+str(num_features)+",num_test_examples="+str(num_test_examples))
#calculate the means of all seen classes
	Xmean_seen = np.zeros(shape=(num_seen_class,num_features))
	for i in range(0,num_seen_class):
		num_examples = np.shape(X_seen[i])[0]
		Xmean_seen[i] = (np.sum(X_seen[i],0)) / (num_examples)
	# print("The dimension of Xmean_seen is "+str(Xmean_seen.shape))
#load class attributes
	class_attributes_seen=np.load('class_attributes_seen.npy')
	class_attributes_unseen=np.load('class_attributes_unseen.npy')
	num_attributes = np.shape(class_attributes_unseen)[1]
	num_unseen_classes = np.shape(class_attributes_unseen)[0]
	# print("The parameters : num_unseen_classes="+str(num_unseen_classes)+",num_attributes="+str(num_attributes))
#calculate the similarity matrix for each unseen class w.r.t the seen classes
	similarity_matrix = np.zeros(shape=(num_unseen_classes,num_seen_class))
	for i in range(0,num_unseen_classes):
		similarity_matrix[i] = LA.norm(class_attributes_seen-class_attributes_unseen[i],axis=1)
	similarity_matrix = similarity_matrix / np.sum(similarity_matrix,1)[:,np.newaxis]
	# print("The dimension of similarity matrix is "+str(similarity_matrix.shape))
#generate means for unseen classes
	Xmean_unseen = np.zeros(shape=(num_unseen_classes,num_features))
	for i in range(0,num_unseen_classes):
			Xmean_unseen[i] = np.sum((Xmean_seen*((similarity_matrix[i])[:,np.newaxis])),axis=0)
	# print("The dimension of Xmean_unseen is "+str(Xmean_unseen.shape))
#assign classes to the test examples using the generated means for unseen classes (prototype model)
	Ypredicted = np.zeros(shape=(num_test_examples,1))
	positive = 0
	for i in range(0,num_test_examples):
		Ypredicted[i] = np.argmin(LA.norm(Xmean_unseen-Xtest[i],axis=1))+1
		#calculate accuracy of the model
		if Ypredicted[i] == Ytest[i]:
			positive += 1
	#print(Ypredicted)
	print("The test accuracy is : " + str(positive/num_test_examples)+" ==> " + str(Ypredicted.shape))
