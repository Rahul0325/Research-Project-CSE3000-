#!/usr/bin/env python
# coding: utf-8

# # Some notes 
# 1) Does contains randomization \
# 2) Runs on 1M dataset 

# In[ ]:





# # Step 0: Imports
# 
# 

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# # Step1: Load data 

# In[3]:


# #Rating information
# r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
# ratings = pd.read_csv('ml-100k/u.data',  sep='\t', names=r_cols, encoding='latin-1')
# ratings.head()

# ------------------------------- Uncomment code below to import 1M dataset -------------------------------------------
#Rating information
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')
ratings.head()


# In[4]:


#User information 
# u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
# users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
# users


# # Step2: Define PIP similarity measure

# In[5]:


rating_max = 5
rating_min = 0
rating_median = (rating_max-rating_min)/2


# ### Agreement function
# The agreement function takes ratings from 2 different users for the SAME movie. \
# If both ratings of the users are on the same sides of the median rating, then return True. \
# else return False.

# In[6]:


def agreement(rating1: int, rating2: int) -> bool:
    if ((rating1 > 2.5 and rating2 < 2.5) or (rating1 < 2.5 and rating2 > 2.5)):
        return False 
    else:
        return True 


# ### Proximity function 
# The proximity function takes the absolute distance between ratings from 2 users for the same movie. \
# Futhermore, if the 2 ratings are in dis-agreement, then a penalty is given to the ratings. \
# the penalty is given by doubling the distance between the ratings, which is then squared (see details in
# 
# If the ratings are in agreement, then no penalty is applied. 

# In[7]:


def proximity(rating1: int, rating2: int) -> float: 
    if(agreement(rating1, rating2)):
        dist = np.absolute(rating1 - rating2)
    else: 
        dist = 2 * np.absolute(rating1 - rating2)
    prox = ((2*(rating_max - rating_min) + 1) - dist) ** 2
    return prox


# ### Impact function
# The inpact function assesses how strongly a movie is liked/disliked by users. \ 
# 
# When 2 users who are in agreement give extreme ratings (like 5's or 0's) to a movie, we can give greater credability to the similarity between these users. \

# In[8]:


def impact(rating1: int, rating2: int) -> float: 
    impact = (np.absolute(rating1 - rating_median) + 1) * (np.absolute(rating2 - rating_median) + 1)
    if(agreement(rating1, rating2)):
        return impact
    else: 
        return 1/impact 


# ### Popularity function
# The Popularity factor gives bigger value to a similarity for ratings that are further from the average rating of a co-rated item. \
# Let $\mu_k $(mu_k) denote the average rating of item k by all users 

# In[9]:


def popularity(rating1: int, rating2: int, mu_k) -> float: 
    pop = 1
    if((rating1 > mu_k and rating2 > mu_k) or (rating1 < mu_k and rating2 < mu_k)):
        pop = 1 + ((rating1 + rating2)/2 - mu_k)**2
    return pop


# # Step 3: Split raw data into train and test sets 

# In[10]:


# Assign X as the original ratings dataframe and y as the user_id column of ratings.

X = ratings.copy()
y = ratings['user_id']

# Split into training and test datasets, stratified along user_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=200) #Original random_state=42


# # Step 4: Construct userToItemRatingsMatrix 

# In[11]:


df_ratings = X_train.pivot(index='user_id', columns='movie_id', values='rating')
df_ratings


# # Step 5: Data cleaning
# 

# In[12]:


df_ratings_dummy = df_ratings.copy(deep=True)
df_ratings_dummy = df_ratings_dummy.fillna(0)
df_ratings_dummy


# TODO: Normalize the data to have zero mean. 
# TODO: Fill NaN values of a given row i with the mean rating of user i
# df_ratings_dummy = df_ratings_dummy.T.fillna(df_ratings_dummy.mean(axis=1)).T


# # Step 5: Calculate userToUserSim matrix using PIP 

# In[13]:


# # similarity_matrix = df_ratings_dummy.copy(deep=True)


# import time

# start = time.time()
# print("hello")


# # Code to create a DF with the right shape and initializing each cell to 0
# similarity_matrix = cosine_similarity(df_ratings_dummy, df_ratings_dummy)
# similarity_matrix_df = pd.DataFrame(similarity_matrix, index=df_ratings.index, columns=df_ratings.index)
# # for col in similarity_matrix_df.columns:
# #     similarity_matrix_df[col].values[:] = 0
# similarity_matrix_df[:] = 0



# sim_score = 0
# for user1_id, row in similarity_matrix_df.iterrows(): #iterate over rows
#     print("Calculating sim column for user_id: " + str(user1_id))
#     for user2_id, value in row.items():
        
        
#         if(user1_id == user2_id):
# #             print("SAME USER ID'S: " + str(user1_id) + " ,"  +str(user2_id))
#             similarity_matrix_df[user1_id][user2_id] = 1
#         else: 
# #             print("Calculating sim_score for:  " + str(user1_id) + " ,"  +str(user2_id))
#             #Step 1: Get co-rated items between user1 and user 2
#             user1_user2_df = df_ratings.loc[[user1_id, user2_id]]
#             all_rated_items = user1_user2_df[user1_user2_df.columns[~user1_user2_df.isnull().all()]] 
#             co_rated_items = all_rated_items.dropna(axis=1)

#             #Step 2: For each co-rated item, calc the sim using PIP 
#             #Step 3: Loop over each co-rated item/movie, and calculate the sim using user1_rating and user2_rating
#             for movie_id in co_rated_items.columns:
#                 mu_k = df_ratings[movie_id].mean()
#                 user1_rating = co_rated_items[movie_id][user1_id]
#                 user2_rating = co_rated_items[movie_id][user2_id]
                
#                 pip = proximity(user1_rating, user2_rating) * impact(user1_rating, user2_rating) * popularity(user1_rating, user2_rating, mu_k)
#                 sim_score = sim_score + pip

#             similarity_matrix_df[user1_id][user2_id] = sim_score
#             #Step 4: Reset the sim_score 
#             sim_score = 0
        
# #         print("row, column = " + str(rowIndex)+ "," + str(columnIndex) + " Value = " + str(value))
# #         print(value, end="\t")




# end = time.time()
# print(end - start)



# In[14]:


# # Save similarity_matrix_df to CSV file 
# similarity_matrix_df.to_pickle("similarity_matrix_df_100k.pkl")


# In[15]:


# # Read in 100k similarity_matrix from pkl file 
# output = pd.read_pickle("similarity_matrix_df_100k.pkl")
# output

# Read in 1M similarity_matrix from pkl file 
output = pd.read_pickle("1M_sim_matrix/1M_sim_matrix_final_0_6040.pkl")
output


# In[16]:


# The code in this cell basically just replaces the value in all the diagonal cells by the max value in the corresponding column. 

similarity_matrix_df_final = output.copy(deep=True) # TODO: Change this to use the "similarity_matrix_df" which is constructed in step5
for col_index in output.index:
    col_max_value = similarity_matrix_df_final[col_index].max()
    similarity_matrix_df_final[col_index][col_index] = col_max_value + 1
    
similarity_matrix_df_final


# In[17]:


# The code in this cell bascially normalizes each column. 

x = similarity_matrix_df_final.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
similarity_matrix_df_final_scaled = pd.DataFrame(x_scaled)
similarity_matrix_df_final_scaled

# Making sure the rows and columns start from 1 instead of 0
similarity_matrix_df_final_scaled.index = np.arange(1, len(similarity_matrix_df_final_scaled) + 1)
similarity_matrix_df_final_scaled.columns = similarity_matrix_df_final_scaled.columns + 1
similarity_matrix_df_final_scaled


# # Step 6: Define function to predict ratings

# In[18]:


ratings_scores = df_ratings[1]
ratings_scores.dropna()


# index_not_rated = ratings_scores[ratings_scores.isnull()]
# index_not_rated.index


# In[19]:


def calculate_ratings(id_movie, id_user): 
    if id_movie in df_ratings:
        cosine_scores = similarity_matrix_df_final_scaled[id_user] #similarity of id_user with every other user
        ratings_scores = df_ratings[id_movie]      #ratings of every other user for the movie id_movie
        
        #won't consider users who havent rated id_movie so drop similarity scores and ratings corresponsing to np.nan
        
        index_not_rated = ratings_scores[ratings_scores.isnull()].index
        
        ratings_scores = ratings_scores.dropna()
        
        cosine_scores = cosine_scores.drop(index_not_rated)
        
        #calculating rating by weighted mean of ratings and cosine scores of the users who have rated the movie
        if (cosine_scores.sum() == 0):
            ratings_movie = 0
        else: 
            ratings_movie = (ratings_scores.dot(cosine_scores))/cosine_scores.sum()
            
    else:
        # TODO: Find a better default value to return instead of just 2.5 
        return 2.5
    return ratings_movie


# # Step 7: Evaluate performance on test set

# In[20]:


def score_on_test_set():
    user_movie_pairs = zip(X_test['movie_id'], X_test['user_id'])
    predicted_ratings = np.array([calculate_ratings(movie, user) for (movie,user) in user_movie_pairs])
    true_ratings = np.array(X_test['rating'])
    score = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    return score


# In[24]:


# X_test.sort_values(by=['user_id'], ascending=True)
# X_test.loc[X_test['user_id'] ==  237].head(100)

def simulate_cold_start(maxNumOfRatingsPerUser, X_test):
    # Step 1: Create an empty dataFrame. 
    X_test_simulated = X_test.copy(deep=True)
    X_test_simulated.drop(X_test_simulated.index, inplace=True)
    
    # Step 2: Loop over every single user_id present in X_test 
    for user_id in X_test['user_id'].unique():
        # Step 3: Take only maxNumOfRatingsPerUser and insert those into our empty dataFrame from step1
        new_user_row = X_test.loc[X_test['user_id']==user_id].sample(frac=1)
        X_test_simulated = pd.concat((X_test_simulated,new_user_row.head(maxNumOfRatingsPerUser)),axis=0) 

        #TODO: Randomize this selection of reviews                          
    return X_test_simulated


# X_test_simulated = X_test.copy(deep=True)
# X_test_simulated.drop(X_test_simulated.index, inplace=True)
# X_test_simulated

# user_id = 1
# user_row = X_test.loc[X_test['user_id']==user_id]
# user_row.sample(frac=1)


# In[25]:


def score_on_test_set_cold_start(simulated_cold_start_test_users, ):
    user_movie_pairs = zip(simulated_cold_start_test_users['movie_id'], simulated_cold_start_test_users['user_id'])
    predicted_ratings = np.array([calculate_ratings(movie, user) for (movie,user) in user_movie_pairs])
    true_ratings = np.array(simulated_cold_start_test_users['rating'])
    score = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    return score


# In[28]:


RMSE_values = np.zeros(60)
for i in range(1, 60):
    simulated_cold_start_test_users = simulate_cold_start(i, X_test)
    score = score_on_test_set_cold_start(simulated_cold_start_test_users)
    print("Now testing for cold-start conditions with MaxRatingsPerUser = " + str(i))
    print(score)
    RMSE_values[i] = score


# In[29]:


# import matplotlib.pyplot as plt
# x_axis = np.array(range(1, 3))
# y_axis = RMSE_values 
# plt.plot(x_axis, y_axis[1:])  # Plot the chart
# plt.xlabel("Cold-start number of reviews per user -> ")
# plt.ylabel("RMSE")
# plt.show()  # display


# In[30]:


## convert your array into a dataframe
df = pd.DataFrame(RMSE_values)

## save to xlsx file

filepath = 'RMSE_PIP_1M_randomization_1.xlsx'

df.to_excel(filepath, index=False)


# In[73]:


X_train['user_id'].unique().shape


# In[76]:





# In[ ]:




