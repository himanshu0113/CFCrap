
#1M dataset function call
def cf_1M():
    print "1M"
    kfold_1M()

#1M dataset kfold function
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold

def kfold_1M():
    rating_file = "ml-1m/ratings.dat"
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    dataset = pd.read_csv(rating_file, sep='::', names = header, engine = 'python')
    users = 6040
    items = 3952
    
    rmin = 1
    rmax = 5
    
    data = dataset.values
    X = data[:, 0:2]
    Y = data[:, 3]
    val_size = 0.30
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=val_size, random_state=seed)
    
    #user-item matrix, train and test matrices
    train_matrix = np.zeros((users, items))
    for r in range(X_train.shape[0]):
        train_matrix[X_train[r][0] -1][X_train[r][1] -1] = Y_train[r]

    test_matrix = np.zeros((users, items))
    for r in range(X_validation.shape[0]):
        test_matrix[X_validation[r][0] -1][X_validation[r][1] -1] = Y_validation[r]
        
    #cosine similarity
    from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
    user_similarity = pairwise_distances(train_matrix, metric = 'cosine')
    item_similarity = pairwise_distances(train_matrix.T, metric = 'cosine')
    
    user_prediction = predict_knn(train_matrix, user_similarity, k = 10, type = 'user')
    item_prediction = predict_knn(train_matrix, item_similarity, k = 10, type = 'item')
    
    from sklearn.metrics import mean_absolute_error as mae

    def nmae(prediction, ground_truth):
        #print prediction[ground_truth.nonzero()].flatten()
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        error = mae(ground_truth, prediction)
        #print error
        return error/(rmax-rmin)
    
    
    print 'User-based CF NMAE \t' + str(nmae(user_prediction, test_matrix))
    print 'Item-based CF NAME \t' + str(nmae(item_prediction, test_matrix))

#100k dataset function call
def cf_100k():
    for i in range(1, 6):
        print '\nResults for Fold ' + str(i) + ' :'
        kfold_100k(i)

#making prediction
def predict(ratings, similarity, type = "user"):
    if type == "user":
        #print ratings
        mean_user_rating = ratings.mean(axis = 1)
        #print mean_user_rating
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:,np.newaxis] + similarity.dot(ratings_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
        np.around(pred, out=pred)

    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        np.around(pred, out=pred)
    return pred

#making prediction
def predict_knn(ratings, similarity, knn, type = "user"):
    if type == "user":
        k=10
        #print ratings
        #mean_user_rating = ratings.mean(axis = 1)
        #print mean_user_rating
        #ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        #pred = mean_user_rating[:,np.newaxis] + similarity[:,:][knn_users[:][:10]].dot(ratings_diff[:, :][knn_users[:][:10]])/np.array([np.abs(similarity[:, :][knn_users[:][:10]]).sum(axis=1)]).T
        pred = np.zeros(ratings.shape)
        for i in range(n_users):
            #top_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            top_users = [knn[i][:]]
            #print top_users
            #print top_users
            for j in range(n_items):
                pred[i, j] = similarity[i, :][top_users].dot(ratings[:, j][top_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_users]))

        np.around(pred, out=pred)

    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        np.around(pred, out=pred)
    return pred

#significance weighing
#corelation
def sigw(train_mat, sim):
    weight = np.zeros((n_users, n_users))
    
    for u in range(n_users):
        for k in range(n_users):
            count = 0
            for i in range(n_items):
                if(train_mat[u][i]>0 and train_mat[k][i]>0):
                    count = count+1
            if count > 30:
                weight[u][k] = 1.0
            else:
                weight[u][k] = 1/30.0

    #user_sim = np.zeros((n_users, n_users))
    user_sim = np.multiply(sim, weight)
    print user_sim

    """
    for u in range(0, n_users):
        for k in range(0, n_users):
            if u == k:
                user_sim[u][k] = -1
            else:
                sum_rr = sum(np.multiply(train_mat[u][:], train_mat[k][:]))
                sum_ruj = sum(np.multiply(train_mat[u][:], train_mat[u][:]))
                sum_rkj = sum(np.multiply(train_mat[k][:], train_mat[k][:]))
                user_sim[u][k] = (sum_rr*weight[u][k])/((sum_ruj**(1.0/2))*(sum_rkj**(1.0/2)))
    """
    return user_sim

#cosine, and variance weighing
def varw(train_mat):
    #varience of items
    mean_item_rating = train_mat.mean(axis = 0)
    
    #mean user ratings
    mean_user_rating = train_mat.mean(axis = 1)
    #print mean_item_rating
    #item_var = [sum([(xur - mean_item_rating[xi])**2 for xi in range(0, 1682)])/ 1682 for xur in train_matrix[:, xi]]

    item_var = np.zeros(n_items)
    for xi in range(0, n_items):
        sum_d = 0
        for xur in train_mat[:, xi]:
            sum_d = sum_d + (xur - mean_item_rating[xi])**2
        item_var[xi] = sum_d/1682

    item_var_sum = sum(item_var)
    #print item_var

    user_sim = np.zeros((n_users, n_users))

    for u in range(0, n_users):
        for k in range(0, n_users):
            if u == k:
                user_sim[u][k] = -1
            else:
                sum_rr = sum(np.multiply(np.multiply(np.subtract(train_mat[u][:], mean_user_rating[u]), np.subtract(train_mat[k][:], mean_user_rating[k])), item_var[:]))
                user_sim[u][k] = sum_rr/item_var_sum
    return user_sim

#KNearest Neighbors and threshold
def knn(similarity, th, k = 21):
    rankings = np.zeros((n_users, n_users))

    for i in range(n_users):
        rankings[i]=np.argsort(similarity[i])

    knn_users = []
    for i in range(n_users):
        knn_users.append([int(x) for x in rankings[i][:k] if x > th])
        #print knn_users[i]
    
    knn_th = []
    for i in range(n_users):
        knn_th.append([int(x) for x in rankings[i][:k] if x > th])

    return knn_users, knn_th



#100k dataset kfold function
def kfold_100k(fold):
    loc_train = "ml-100k/u" + str(fold) + ".base"
    loc_test = "ml-100k/u" + str(fold) + ".test"
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    train_data = pd.read_csv(loc_train, sep = '\t', names = header)
    test_data = pd.read_csv(loc_test, sep = '\t', names = header)
    #print (train_data)

    #rmin = min(train_data.rating)
    #rmax = max(train_data.rating)
    rmin = 1
    rmax = 5

    #user-item matrix, train and test matrices
    train_matrix = np.zeros((n_users, n_items))
    for record in train_data.itertuples():
        train_matrix[record[1] -1][record[2] -1] = record[3]

    test_matrix = np.zeros((n_users, n_items))
    for record in test_data.itertuples():
        test_matrix[record[1] -1][record[2] -1] = record[3]
        
    #cosine similarity
    from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
    user_similarity = pairwise_distances(train_matrix, metric = 'cosine')
    #item_similarity = pairwise_distances(train_matrix.T, metric = 'cosine')
    
    #call to calculate user similarity
    #user_similarity = pearson_sim(train_matrix, np.divide(item_var, item_var_sum))
    
    #user_similarity = sigw(train_matrix)
    #user_similarity = varw(train_matrix)
    
    from sklearn.metrics import mean_absolute_error as mae

    def nmae(prediction, ground_truth):
        #print prediction[ground_truth.nonzero()].flatten()
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        error = mae(ground_truth, prediction)
        #print error
        return error/(rmax-rmin)
    

    user_similarity = sigw(train_matrix, user_similarity)
    threshold = 0.6
    k = 21
    knn_users, knn_th = knn(user_similarity, threshold, k)
    user_prediction = predict_knn(train_matrix, user_similarity, knn_users, type = 'user')
    print "User-based CF NMAE with k = " + str(k) + " : " +  str(nmae(user_prediction, test_matrix))
    
    k = 41
    knn_users, knn_th = knn(user_similarity, threshold, k)
    user_prediction = predict_knn(train_matrix, user_similarity, knn_users, type = 'user')
    print "User-based CF NMAE with k = " + str(k) + " : " +  str(nmae(user_prediction, test_matrix))
    
    k = 61
    knn_users, knn_th = knn(user_similarity, threshold, k)
    user_prediction = predict_knn(train_matrix, user_similarity, knn_users, type = 'user')
    print "User-based CF NMAE with k = " + str(k) + " : " +  str(nmae(user_prediction, test_matrix))
    
    k = 81
    knn_users, knn_th = knn(user_similarity, threshold, k)
    user_prediction = predict_knn(train_matrix, user_similarity, knn_users, type = 'user')
    print "User-based CF NMAE with k = " + str(k) + " : " +  str(nmae(user_prediction, test_matrix))
    
    
    """"
    item_prediction = predict(train_matrix, item_similarity, type = 'item')
    print 'Item-based CF NAME \t' + str(nmae(item_prediction, test_matrix))
    
    user_similarity = varw(train_matrix)
    user_prediction = predict_knn(train_matrix, user_similarity, k = 10, type = 'user')
    print 'Variance Weighting \t' + str(nmae(user_prediction, test_matrix))
    
    
    user_similarity = sigw(train_matrix, user_similarity)
    user_prediction = predict_knn(train_matrix, user_similarity, k = 10, type = 'user')
    sigw_nmae = nmae(user_prediction, test_matrix)
    print 'Significance Weighting \t' + str(sigw_nmae)
    
    user_prediction = predict_knn(train_matrix, user_similarity, k = 20, type = 'user')
    sigw_nmae2 = nmae(user_prediction, test_matrix)
    print 'Significance Weighting \t' + str(sigw_nmae2)
    """
    """
    # user-item model
    lamda = 0.6
    ui_prediction = np.add(np.multiply(lamda,user_prediction), np.multiply(1-lamda,item_prediction))
    print 'User-Item based CF NMAE: ' + str(nmae(ui_prediction, test_matrix))
    """
    

#CF Assignment

import numpy as np
import pandas as pd

np.set_printoptions(threshold=10)

n_users = 943
n_items = 1682

option = input("Select dataset : \n 1. 100k Movie-lens\n 2. 1M Movie-lens\n")

if option==1:
    print 'Results on Movielens 100K dataset : \n'
    cf_100k()
else:
    cf_1M() 

