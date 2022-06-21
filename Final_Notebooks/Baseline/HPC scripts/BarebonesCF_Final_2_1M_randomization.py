#!/usr/bin/env python
# coding: utf-8

# # Some notes:
# 1) Does not contain raindomization. \
# 2) Runs on 1M data-set

# # Step 0: Imports
# 
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from surprise import Reader, Dataset, KNNBasic
# from surprise.model_selection import cross_validate
# from surprise import SVD


# # Step 1: Import data

# In[2]:


# r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
# ratings = pd.read_csv('ml-100k/u.data',  sep='\t', names=r_cols, encoding='latin-1')
# ratings.head()

#Rating information
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')
ratings.head()


# In[3]:


# ---------------------------------------------- I never even use the movies table --------------------------------------
# i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
# movies = pd.read_csv('ml-100k/u.item',  sep='|', names=i_cols, encoding='latin-1')
# movies.head()


# In[4]:


# u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
# users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
# users


# ------------------------------- Uncomment code below to import 1M dataset -------------------------------------------
unames = ['user_id','sex','age','occupation','zip']
# users1M = pd.read_csv('ml-1m/users.dat', sep='::', names=unames, encoding='latin-1')
users = pd.read_table('ml-1m/users.dat', sep='::', header=None, names=unames, engine='python')
users


# # Step 2: Split raw data into train and test sets 

# In[5]:


# Assign X as the original ratings dataframe and y as the user_id column of ratings.

X = ratings.copy()
y = ratings['user_id']

# Split into training and test datasets, stratified along user_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=200) #Original random_state=42


# # Step 3: Construct userToItemRatings matrix
# 

# In[6]:


df_ratings = X_train.pivot(index='user_id', columns='movie_id', values='rating')
# df_ratings


# # Step 4: Data cleaning
# 

# In[7]:


df_ratings_dummy = df_ratings.copy().fillna(0) 
df_ratings_dummy.head()

# TODO: Normalize the data to have zero mean. 
# TODO: Fill NaN values of a given row i with the mean rating of user i


# # Step 5: Calculate userToUserSim matrix
# 

# In[8]:


similarity_matrix = cosine_similarity(df_ratings_dummy, df_ratings_dummy)
similarity_matrix_df = pd.DataFrame(similarity_matrix, index=df_ratings.index, columns=df_ratings.index)
similarity_matrix_df


# # Step 6: Define function to predict ratings
# 

# In[9]:


ratings_scores = df_ratings[1]
ratings_scores.dropna()


# index_not_rated = ratings_scores[ratings_scores.isnull()]
# index_not_rated.index


# In[10]:


def calculate_ratings(id_movie, id_user): 
    if id_movie in df_ratings:
        cosine_scores = similarity_matrix_df[id_user] #similarity of id_user with every other user
        ratings_scores = df_ratings[id_movie]      #ratings of every other user for the movie id_movie
        
        #won't consider users who havent rated id_movie so drop similarity scores and ratings corresponsing to np.nan
        
        index_not_rated = ratings_scores[ratings_scores.isnull()].index
        
        ratings_scores = ratings_scores.dropna()
        
        cosine_scores = cosine_scores.drop(index_not_rated)
        
        #calculating rating by weighted mean of ratings and cosine scores of the users who have rated the movie
        if(cosine_scores.sum() == 0): 
            ratings_movie = 0
        else: 
            ratings_movie = np.dot(ratings_scores, cosine_scores)/cosine_scores.sum()
    
    else:
        # TODO: Find a better default value to return instead of just 2.5 
        return 2.5
    return ratings_movie


# In[11]:


# calculate_ratings(3,150) #predicts rating for user_id 150 and movie_id 3


# # Step 7: Evaluate performance on test set
# 

# In[12]:


def score_on_test_set():
    user_movie_pairs = zip(X_test['movie_id'], X_test['user_id'])
    predicted_ratings = np.array([calculate_ratings(movie, user) for (movie,user) in user_movie_pairs])
    true_ratings = np.array(X_test['rating'])
    score = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    return score


# In[13]:


# How to find the unique values from a particular column 
# X_test['user_id'].unique()

# for user_id in X_test['user_id'].unique():
#     # Step 3: Take only maxNumOfRatingsPerUser 
#     print("considering user_id: " + str(user_id))
#     print(X_test.loc[X_test['user_id']==user_id].head(20))
# So now we have all the unique user_id's in X_test


# In[14]:


# X_test.sort_values(by=['user_id'], ascending=True)
# X_test.loc[X_test['user_id'] ==  237].head(100)

# X_test_simulated = X_test.copy(deep=True)
# X_test_simulated.drop(X_test_simulated.index, inplace=True)
# X_test_simulated

def simulate_cold_start(maxNumOfRatingsPerUser, X_test):
    # Step 1: Create an empty dataFrame. 
    X_test_simulated = X_test.copy(deep=True)
    X_test_simulated.drop(X_test_simulated.index, inplace=True)
    
    # Step 2: Loop over every single user_id present in X_test 
    for user_id in X_test['user_id'].unique():
        # Step 3: Take only maxNumOfRatingsPerUser and insert those into our empty dataFrame from step1
        new_user_row = X_test.loc[X_test['user_id']==user_id].sample(frac=1)
        X_test_simulated = pd.concat((X_test_simulated,new_user_row.head(maxNumOfRatingsPerUser)),axis=0) 
        
    return X_test_simulated


# In[15]:


def score_on_test_set_cold_start(simulated_cold_start_test_users, ):
    user_movie_pairs = zip(simulated_cold_start_test_users['movie_id'], simulated_cold_start_test_users['user_id'])
    predicted_ratings = np.array([calculate_ratings(movie, user) for (movie,user) in user_movie_pairs])
    true_ratings = np.array(simulated_cold_start_test_users['rating'])
    score = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    return score


# In[16]:


import time
start_time = time.time()

RMSE_values = np.zeros(60)
for i in range(1, 60):
    simulated_cold_start_test_users = simulate_cold_start(i, X_test)
    score = score_on_test_set_cold_start(simulated_cold_start_test_users)
    print("Now testing for cold-start conditions with MaxRatingsPerUser = " + str(i))
    print(score)
    RMSE_values[i] = score
    
print("--- %s seconds ---" % (time.time() - start_time))


# In[17]:


# import matplotlib.pyplot as plt
# x_axis = np.array(range(1, 60))
# y_axis = RMSE_values 
# plt.plot(x_axis, y_axis[1:])  # Plot the chart
# plt.xlabel("Cold-start number of reviews per user -> ")
# plt.ylabel("RMSE")
# plt.show()  # display


# In[18]:


# x_axis = np.array(range(1, 21))
# y_axis = RMSE_values 
# plt.plot(x_axis, y_axis[1:21])  # Plot the chart
# plt.xlabel("Cold-start number of reviews per user -> ")
# plt.ylabel("RMSE")
# plt.show()  # display


# In[20]:


## convert your array into a dataframe
df = pd.DataFrame(RMSE_values)

## save to xlsx file

filepath = 'baseline_1M_iteration1.xlsx'

df.to_excel(filepath, index=False)


# In[ ]:




