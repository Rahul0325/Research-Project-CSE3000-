#!/usr/bin/env python
# coding: utf-8

# # Step 0: Imports and loading user data
# 
# 

# In[2]:


import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
# users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
# users


# ------------------------------- Uncomment code below to import 1M dataset -------------------------------------------
unames = ['user_id','sex','age','occupation','zip']
# users1M = pd.read_csv('ml-1m/users.dat', sep='::', names=unames, encoding='latin-1')
users = pd.read_table('ml-1m/users.dat', sep='::', header=None, names=unames, engine='python')
users
# users.isna().sum() # Shows that no rows contain any features that have null values. 


# In[4]:


users_without_zip = users.loc[:, ['age', 'sex']]
users_without_zip
# users_without_zip.loc[users_without_zip['sex'] == 'M']


# # Step1:  Encode the categorical features 

# In[5]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# In[6]:


col_trans = make_column_transformer((OneHotEncoder(), ['sex']), remainder='passthrough')


# In[7]:


col_trans.fit_transform(users_without_zip)
dummy_users = pd.DataFrame(col_trans.fit_transform(users_without_zip))
dummy_users


# # Step 2: Scale each feature

# In[8]:


from sklearn.preprocessing import MinMaxScaler 


# In[9]:


scaled_dummy_users = dummy_users.copy(deep=True)
scaler = MinMaxScaler()
scaler.fit(dummy_users[[2]])
scaled_dummy_users[2] = scaler.transform(dummy_users[[2]])
scaled_dummy_users


# # Step 3: Apply k-means clustering

# In[10]:


from sklearn.cluster import KMeans
import sklearn.metrics as metrics


# In[11]:


k_range = range(1,10)
sse = []
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(scaled_dummy_users)
    sse.append(km.inertia_)


# In[12]:


sse


# In[13]:


plt.xlabel("k")
plt.ylabel("sim of squared error")
plt.plot(k_range, sse)


# In[14]:


for i in range(2, 20):
    labels = KMeans(n_clusters=i, init='k-means++', random_state=200).fit(scaled_dummy_users).labels_
    score = metrics.silhouette_score(scaled_dummy_users, labels, metric="euclidean", sample_size=1000, random_state=200)
    print("Silhouette score for k(clusters) = " + str(i) + " is " +   str(score))


# In[15]:


kmeans = KMeans(n_clusters=2, init='k-means++', random_state=200).fit(scaled_dummy_users)
cluster = kmeans.labels_
clusters_scaled_dummy_users = scaled_dummy_users.copy(deep=True)
clusters_scaled_dummy_users['clusters'] = cluster
clusters_scaled_dummy_users['user_id'] = users['user_id'] # Adding back the user_id's for each user. 
clusters_scaled_dummy_users


# In[16]:


for i in range(0, 7):
    cluster_size = len(clusters_scaled_dummy_users.loc[clusters_scaled_dummy_users['clusters'] == i])
    print("Cluster " + str(i) + " : " + str(cluster_size))


# In[ ]:





# In[17]:


kmeans.predict(scaled_dummy_users.loc[[941]])


# # Phase 2: Collaborative filtering
# ## Step 0: Imports

# In[18]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ## Step 1: Import data
# 

# In[19]:


# r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
# ratings = pd.read_csv('ml-100k/u.data',  sep='\t', names=r_cols, encoding='latin-1')
# ratings.head()


# ------------------------------- Uncomment code below to import 1M dataset -------------------------------------------
#Rating information
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')
ratings.head()


# In[20]:


# i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
# movies = pd.read_csv('ml-100k/u.item',  sep='|', names=i_cols, encoding='latin-1')
# movies.head()


# ------------------------------- Uncomment code below to import 1M dataset -------------------------------------------
#Movie information
# movies = pd.read_csv('ml-1m/movies.dat', engine='python', sep='::', names=['movieid', 'title', 'genre']).set_index('movieid')
# movies['genre'] = movies.genre.str.split('|')


# ## Step 2: Split raw data into train and test sets 

# In[21]:


# Assign X as the original ratings dataframe and y as the user_id column of ratings.

X = ratings.copy()
y = ratings['user_id']

# Split into training and test datasets, stratified along user_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=200) #Original random_state=42


# ## Step 3: Construct userToItemRatings matrix

# In[22]:


df_ratings = X_train.pivot(index='user_id', columns='movie_id', values='rating')
df_ratings


# ## Step 4: Data cleaning

# In[23]:


df_ratings_dummy = df_ratings.copy().fillna(0) 
df_ratings_dummy.head()

# TODO: Normalize the data to have zero mean. 
# TODO: Fill NaN values of a given row i with the mean rating of user i


# # Step 5: Calculate userToUserSim matrix
# 

# In[24]:


similarity_matrix = cosine_similarity(df_ratings_dummy, df_ratings_dummy)
similarity_matrix_df = pd.DataFrame(similarity_matrix, index=df_ratings.index, columns=df_ratings.index)
similarity_matrix_df


# ## Step 6: Define function to predict ratings
# 

# In[55]:


def calculate_ratings(id_movie, id_user): 
    if id_movie in df_ratings:
        cosine_scores = similarity_matrix_df[id_user] #similarity of id_user with every other user
        ratings_scores = df_ratings[id_movie]      #ratings of every other user for the movie id_movie

        #won't consider users who havent rated id_movie so drop similarity scores and ratings corresponsing to np.nan

        index_not_rated = ratings_scores[ratings_scores.isnull()].index

        ratings_scores = ratings_scores.dropna()

        cosine_scores = cosine_scores.drop(index_not_rated)


# ---------------------------------------------------------------------

        # Find the user-id's of users who are in the same/different cluster
        cluser_of_user = clusters_scaled_dummy_users.loc[clusters_scaled_dummy_users['user_id'] == id_user]['clusters'].item()
        # cluser_of_user

        users_in_other_cluster = clusters_scaled_dummy_users.loc[clusters_scaled_dummy_users['clusters'] != cluser_of_user]
        # users_in_other_cluster
        index_of_users_in_other_clusters = users_in_other_cluster['user_id'].values[:]
        # index_of_users_in_other_clusters

        users_in_same_cluster = clusters_scaled_dummy_users.loc[clusters_scaled_dummy_users['clusters'] == cluser_of_user]
        index_of_users_in_same_clusters = users_in_same_cluster['user_id'].values[:]
        # index_of_users_in_same_clusters

        cosine_scores_same_cluster = cosine_scores.drop(index_of_users_in_other_clusters, errors='ignore')
        # cosine_scores_same_cluster
        cosine_scores_other_cluster = cosine_scores.drop(index_of_users_in_same_clusters, errors='ignore')
        # cosine_scores_other_cluster 
        
        ratings_scores_same_cluster = ratings_scores.drop(index_of_users_in_other_clusters, errors='ignore')
        ratings_scores_other_cluster = ratings_scores.drop(index_of_users_in_same_clusters, errors='ignore')

# --------------------------------------------------------------------------------       
        
        if(cosine_scores_other_cluster.sum() == 0 or cosine_scores_same_cluster.sum() == 0): 
            #calculating rating by weighted mean of ratings and cosine scores of the users who have rated the movie
            if (cosine_scores.sum() == 0):
                ratings_movie = 0
            else: 
                ratings_movie = (ratings_scores.dot(cosine_scores))/cosine_scores.sum()
        else: 
            ratings_movie_same_cluster = (ratings_scores_same_cluster.dot(cosine_scores_same_cluster))/cosine_scores_same_cluster.sum()
            ratings_movie_other_cluster = (ratings_scores_other_cluster.dot(cosine_scores_other_cluster))/cosine_scores_other_cluster.sum()
            ratings_movie = 0.8*ratings_movie_same_cluster + 0.2*ratings_movie_other_cluster
    
    else:
        # TODO: Find a better default value to return instead of just 2.5 
        return 2.5
    return ratings_movie


# In[ ]:





# In[56]:


# calculate_ratings(3,150) #predicts rating for user_id 150 and movie_id 3


# ## Step 7: Evaluate performance on test set
# 

# In[57]:


def simulate_cold_start(maxNumOfRatingsPerUser, X_test):
    # Step 1: Create an empty dataFrame. 
    X_test_simulated = X_test.copy(deep=True)
    X_test_simulated.drop(X_test_simulated.index, inplace=True)
    
    # Step 2: Loop over every single user_id present in X_test 
    for user_id in X_test['user_id'].unique():
        # Step 3: Take only maxNumOfRatingsPerUser and insert those into our empty dataFrame from step1
        X_test_simulated = X_test_simulated.append(X_test.loc[X_test['user_id']==user_id].head(maxNumOfRatingsPerUser), ignore_index=True)
        
    return X_test_simulated


# In[58]:


def score_on_test_set_cold_start(simulated_cold_start_test_users, ):
    user_movie_pairs = zip(simulated_cold_start_test_users['movie_id'], simulated_cold_start_test_users['user_id'])
    predicted_ratings = np.array([calculate_ratings(movie, user) for (movie,user) in user_movie_pairs])
    true_ratings = np.array(simulated_cold_start_test_users['rating'])
    score = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    return score


# In[59]:


RMSE_values = np.zeros(60)
for i in range(1, 60):
    simulated_cold_start_test_users = simulate_cold_start(i, X_test)
    score = score_on_test_set_cold_start(simulated_cold_start_test_users)
    print("Now testing for cold-start conditions with MaxRatingsPerUser = " + str(i))
    print(score)
    RMSE_values[i] = score


# In[40]:


# id_user = 1
# id_movie = 1

# cosine_scores = similarity_matrix_df[id_user] #similarity of id_user with every other user
# ratings_scores = df_ratings[id_movie]      #ratings of every other user for the movie id_movie
        
# #won't consider users who havent rated id_movie so drop similarity scores and ratings corresponsing to np.nan
        
# index_not_rated = ratings_scores[ratings_scores.isnull()].index
        
# ratings_scores = ratings_scores.dropna()
        
# cosine_scores = cosine_scores.drop(index_not_rated)




# # Find the user-id's of users who are in the same/different cluster
# cluser_of_user = clusters_scaled_dummy_users.loc[clusters_scaled_dummy_users['user_id'] == id_user]['clusters'].item()
# # cluser_of_user

# users_in_other_cluster = clusters_scaled_dummy_users.loc[clusters_scaled_dummy_users['clusters'] != cluser_of_user]
# # users_in_other_cluster
# index_of_users_in_other_clusters = users_in_other_cluster['user_id'].values[:]
# # index_of_users_in_other_clusters

# users_in_same_cluster = clusters_scaled_dummy_users.loc[clusters_scaled_dummy_users['clusters'] == cluser_of_user]
# index_of_users_in_same_clusters = users_in_same_cluster['user_id'].values[:]
# # index_of_users_in_same_clusters

# cosine_scores_same_cluster = cosine_scores.drop(index_of_users_in_other_clusters, errors='ignore')
# # cosine_scores_same_cluster
# cosine_scores_other_cluster = cosine_scores.drop(index_of_users_in_same_clusters, errors='ignore')
# # cosine_scores_other_cluster


# ratings_scores_same_cluster = ratings_scores.drop(index_of_users_in_other_clusters, errors='ignore')
# ratings_scores_other_cluster = ratings_scores.drop(index_of_users_in_same_clusters, errors='ignore')


# In[60]:


# import matplotlib.pyplot as plt
# x_axis = np.array(range(1, 60))
# y_axis = RMSE_values 
# plt.plot(x_axis, y_axis[1:])  # Plot the chart
# plt.xlabel("Cold-start number of reviews per user -> ")
# plt.ylabel("RMSE")
# plt.show()  # display


# In[ ]:


print("Now saving the RMSE values as an excel file...")
## convert your array into a dataframe
df = pd.DataFrame(RMSE_values)

## save to xlsx file

filepath = 'baseline_1M_iteration1.xlsx'

df.to_excel(filepath, index=False)
print("Finished saving RMSE values!")


# In[ ]:




