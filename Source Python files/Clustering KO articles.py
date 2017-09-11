from em_utilities import *
import sframe as sf
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.neighbors import NearestNeighbors
import scipy
import time


# # Section 0:
# ## Dataset definition and feature extraction (tf-idf)




dataset= sf.SFrame('Dataset/KO_data.csv')
dataset.remove_column('X1')
dataset= dataset.add_row_number()
dataset.rename({'id':'X1'})




tfidfvec= TfidfVectorizer(stop_words='english')
tf_idf_matrix= tfidfvec.fit_transform(dataset['text'])
tf_idf_matrix = normalize(tf_idf_matrix)


# # Section 1: 
# ## Model Parameters smart initialization
# 
# Used Kmeans++ model to initialize the parameters for the model of EM algorithm.
# - Kmeans++ used to initialize the means (Centroids of clusters)




#Smart Initialization for means with using KMeans++ model 
def initialize_means(num_clusters,features_matrix):
    from sklearn.cluster import KMeans
    np.random.seed(5)
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=5, max_iter=400, random_state=1, n_jobs=1)
    kmeans_model.fit(features_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    means = [centroid for centroid in centroids]
    return [means , cluster_assignment]




#Smart initialization for weights
def initialize_weights(num_clusters,features_matrix,cluster_assignment):
    num_docs = features_matrix.shape[0]
    weights = []
    for i in xrange(num_clusters):
        num_assigned = len(cluster_assignment[cluster_assignment==i]) # YOUR CODE HERE
        w = float(num_assigned) / num_docs
        weights.append(w)
    return weights




#Smart initialization for covariances
def initialize_covs(num_clusters,features_matrix,cluster_assignment):
    covs = []
    for i in xrange(num_clusters):
        member_rows = features_matrix[cluster_assignment==i]
        cov = (member_rows.multiply(member_rows) - 2*member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0]         + means[i]**2
        cov[cov < 1e-8] = 1e-8
        covs.append(cov)
    return covs


# # Section 2:
# ## Training Models with different number of clusters
# 
# Initializing the parameters for each model then start training using the Expectation-Maximization algorithm.




# Model 1 with 10 clusters
(means , cluster_assignment_10model)= initialize_means(10,tf_idf_matrix)
covs= initialize_covs(10,tf_idf_matrix, cluster_assignment_10model)
weights= initialize_weights(10,tf_idf_matrix, cluster_assignment_10model)
model_em_10k= EM_for_high_dimension(tf_idf_matrix, means, covs, weights, cov_smoothing=1e-10)




# Model 2 with 20 clusters.
(means , cluster_assignment_20model)= initialize_means(20,tf_idf_matrix)
covs= initialize_covs(20,tf_idf_matrix, cluster_assignment_20model)
weights= initialize_weights(20,tf_idf_matrix, cluster_assignment_20model)
model_em_20k= EM_for_high_dimension(tf_idf_matrix, means, covs, weights, cov_smoothing=1e-10)


# # Section 3:
# ## Evaluation report for each cluster (Interpreting clusters)
# 
# Evaluation report is divided into two partitions the first one is the word representation for each cluster the really interpret the cluster, the second one is for the variety of article types in one cluster counting each category for each cluster.


def visualize_EM_clusters(tf_idf, means, covs, map_index_to_word):
    print('')
    print('==========================================================')

    num_clusters = len(means)
    for c in xrange(num_clusters):
        print('Cluster {0:d}: Largest mean parameters in cluster '.format(c))
        print('\n{0: <12}{1: <12}{2: <12}'.format('Word', 'Mean', 'Variance'))
        
        # The k'th element of sorted_word_ids should be the index of the word 
        # that has the k'th-largest value in the cluster mean. Hint: Use np.argsort().
        sorted_word_ids = np.argsort(means[c])[::-1]

        for i in sorted_word_ids[:10]:
            print '{0: <12}{1:<10.2e}{2:10.2e}'.format(map_index_to_word[i], 
                                                       means[c][i],
                                                       covs[c][i])
        print '\n=========================================================='



def clusters_report(clusters_idx):
    cluster_id=0
    for cluster_indicies in clusters_idx:
        countP=0
        countB=0
        countE=0
        for i in cluster_indicies:
            if dataset['category'][i]=='product':
                countP+=1
            elif dataset['category'][i]=='engineering':
                countE+=1
            elif dataset['category'][i]=='business':
                countB+=1
        print "Cluster ",cluster_id ,"\n==========================\n"
        cluster_id+=1
        print "product count : ",countP ,"\nengineering count : ",countE,"\nbusiness count : ",countB , "\n"
    




visualize_EM_clusters(tf_idf_matrix, model_em_10k['means'], model_em_10k['covs'], tfidfvec.get_feature_names())



visualize_EM_clusters(tf_idf_matrix, model_em_20k['means'], model_em_20k['covs'], tfidfvec.get_feature_names())


# No. of articles in each cluster for first model with 10 clusters
resps_10k= sf.SFrame(model_em_10k['resp'])
resps_10k= resps_10k.unpack('X1', '')
cluster_id=0
cluster_hash_10model = {}
for col in resps_10k.column_names():
    cluster_10k= np.array(resps_10k[col])
    print "cluster ",cluster_id , "assignments: ", cluster_10k.sum()
    cluster_hash_10model[cluster_id] =cluster_10k.nonzero() 
    cluster_id+=1



# No. of articles in each cluster for second model with 20 clusters
resps_20k= sf.SFrame(model_em_20k['resp'])
resps_20k= resps_20k.unpack('X1', '')
cluster_id=0
cluster_hash_20model = {}
for col in resps_20k.column_names():
    cluster_20k= np.array(resps_20k[col])
    print "cluster ",cluster_id , "assignments: ", cluster_20k.sum()
    cluster_hash_20model[cluster_id] =cluster_20k.nonzero() 
    cluster_id+=1



# Articles' categories in model 1 with 10 clusters
clusters_10k_idx=[]
for col in resps_10k.column_names():
    cluster_10k= np.array(resps_10k[col])
    cluster_10k= cluster_10k.nonzero()[0]
    clusters_10k_idx.append(cluster_10k)
clusters_report(clusters_10k_idx)


# Articles' categories in model 2 with 20 clusters
clusters_20k_idx=[]
for col in resps_20k.column_names():
    cluster_20k= np.array(resps_20k[col])
    cluster_20k= cluster_20k.nonzero()[0]
    clusters_20k_idx.append(cluster_20k)
clusters_report(clusters_20k_idx)


# # Section 4
# ## Recommendation and predictions for Articles
# 
# #### Recommendation method: 
# A method for recommending articles by retrieving the cluster that the article belong to, then fetch all the articles in that cluster articles passed to nearest neighbour model to find the best 10 articles recommended for this article.
# 
# #### Predicting method:
# Sending set of articles to predict the cluster it belong based on the trained data 
# 
# 
# - Using the test dataset to predict cluster for each one using two different models.


def articles_inds(article_id , cluster_hash_model):
    for cluster_id in cluster_hash_model: 
        np_array = np.array(cluster_hash_model[cluster_id])
        if article_id in np_array:
            return cluster_id, np_array



def recommender(article_id ,cluster_hash_model, no_articles, data_articles):
    start_time = time.time()
    cid , inds = articles_inds(article_id ,cluster_hash_model)
    cluster_articles= data_articles.filter_by(inds[0] , 'X1')
    cluster_articles = cluster_articles.add_row_number()

    recom_vec= TfidfVectorizer(stop_words='english')
    tfidf_recommend= recom_vec.fit_transform(cluster_articles['text'])
    tfidf_recommend = normalize(tfidf_recommend)
    
    row_id = cluster_articles[cluster_articles['X1']==article_id]['id'][0]
    NN_model = NearestNeighbors(n_neighbors=no_articles).fit(tfidf_recommend)
    distances, indices = NN_model.kneighbors(tfidf_recommend[row_id])
    
    recommended_ids=[]
    for i in indices[0]:
        recommended_ids.append(cluster_articles[cluster_articles['id']==i]['X1'][0])
    
    del cluster_articles
    del tfidf_recommend
    del recom_vec
    #print("--- %s seconds ---" % (time.time() - start_time))
    #print len(inds[0])
    return recommended_ids


def predict_cluster(articles,em_model):
    article_tfidf= tfidfvec.transform(articles['text'])
    mu= deepcopy(em_model['means'])
    sigma= deepcopy(em_model['covs'])
    assignments=[]
    for j in range(article_tfidf.shape[0]):
        resps=[]
        for i in range(len(em_model['weights'])):
            predict= np.log(em_model['weights'][i]) + logpdf_diagonal_gaussian(article_tfidf[j], mu[i],sigma[i])
            resps.append(predict)
        assignments.append(resps.index(np.max(resps)))
    return assignments



# Recommend articles for all dataset then append it into the SFrame database then export it.
recommended_inds = []
start_time = time.time()
for i in range(len(dataset)):
    recommended_inds.append(recommender(i,cluster_hash_20model,11,dataset))

print("--- %s seconds (Final time complexity): ---" % (time.time() - start_time))


rec_inds= sf.SArray(recommended_inds)
dataset.add_column(rec_inds,name='recommendations')



dataset.save('Articles_with_recommendations.csv',format='csv')




#Saving each cluster data in a seperate CSV file
for cluster_id in cluster_hash_20model:
    ind= np.array(cluster_hash_20model[cluster_id])
    #print ind
    cluster_articles= dataset.filter_by(ind[0] , 'X1')
    cluster_articles.save('Clusters_model20/cluster_'+str(cluster_id)+'.csv',format='csv')
    del cluster_articles


# ### Testing data for cluster assigning.


testset = sf.SFrame('Dataset/KO_articles_test.csv')



test_tfidf= tfidfvec.transform(testset['text'])
# Predict Using model with 10 clusters.
test_predictions= predict_cluster(testset,model_em_10k)
test_predictions= np.array(test_predictions)
test_predictions



# Predict Using model with 20 clusters.
test_predictions= predict_cluster(testset,model_em_20k)
test_predictions= np.array(test_predictions)
test_predictions
	
