
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import  fetch_openml
import operator



mnist = fetch_openml("mnist_784", as_frame=False)
data = mnist["data"]
labels = mnist["target"]
idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:1000], :].astype(int)
train_labels = labels[idx[:1000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]




"""
 
@:param train: a set of train images; 
@:param labels: a vector of labels, corresponding to the images;
@:param query_img: a query image; 
@:param k: a number k. The function wil


implement the k-NN algorithm to return a prediction of the query image, given the train
images and labels. The function will use the k nearest neighbors, using the Euclidean
L2 metric. In case of a tie between the k labels of neighbors, it will choose an arbitrary
option.

"""
def classifyer(query_img, train, labels, k,q3=False):
    distances = dist(train, query_img)
    sortedDistIndices = distances.argsort()
    classCount = {}
    best_for_eachK={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        best_for_eachK[i] =sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    if q3:
        return best_for_eachK
    return sortedClassCount[0][0]

def classifyer_for_n_querys(n):
    num_correct = 0
    for i in range(1, n):
        if test_labels[i] == classifyer(test[i], train, train_labels, 1):
            num_correct += 1
    accuracy = float(num_correct) / n
    print("******** section (b) ******** ")
    print('Got %d / %d correct => accuracy: %f' % (num_correct, n, accuracy))
    return 0



def classifyer_for_n_querys_with_k(n, k,q3 = False,q4=False):
    global hit_per_k
    global last_index


    if q3:
        hit_per_k = np.zeros(k)
        for i in range(1, n):
            temp=classifyer(test[i], train, train_labels, k, q3=q3)
            for j in range(1,k):
                if test_labels[i]  == temp[j][0][0] :
                    hit_per_k[j]+=1

    best_K=np.amax(hit_per_k)
    accuracy = float(best_K) / n
    print("******** section (c) ******** ")
    print('best K found :  k=' , np.where(hit_per_k == np.amax(hit_per_k))[0][0]+1,' with %d / %d correct => accuracy: %f' % (best_K, n, accuracy))
    return hit_per_k/n

def section_b():
    classifyer_for_n_querys(1000)

def section_c():
    k_to_accuracies = (classifyer_for_n_querys_with_k(1000, 100, q3=True))
    k_choices = np.linspace(1, 100, 100, dtype=int)
    for k in range(1, 99):
        accuracies = k_to_accuracies[k]
        plt.scatter([k], accuracies)
    plt.plot(k_to_accuracies, k_choices, color='b')
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylim([0.68, 0.89])
    plt.ylabel('Cross-validation accuracy')
    plt.show()

def dist(set_train, image):
    dist_to_return = np.zeros(len(set_train))
    for i in range(len(set_train)):
        dist_to_return[i] = np.linalg.norm(set_train[i] - image)
    return dist_to_return

if __name__ == '__main__':

         """  Section (b)  """

         section_b()

         """  Section (c)  """

         section_c()

         """  Section (d)  """
         d=np.linspace(100, 5000, 50)
         cnt=0
         best_n=np.zeros(51)






















