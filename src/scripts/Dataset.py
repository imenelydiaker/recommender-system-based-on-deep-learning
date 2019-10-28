import pandas as pn
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def parse_args() :
    #DEFAULT PATHS TO CHANGE
    parser = argparse.ArgumentParser(description = "Datasets Processing.")
    parser.add_argument('--type', type = int, default= 2,
                        help= 'Type of spliting the dataset:\n 1)Test with new users   2)Normal test set')
    parser.add_argument('--dataset', type = int, default= 1,
                        help = 'Choose a dataset:\n 1)MovieLens 100K  2)MovieLens 1M  3)Yelp.')
    parser.add_argument('--negs', type = int, default= 1,
                        help= 'Number of negatives per positive instance.')
    parser.add_argument('--path_save', nargs = '?', default= 'D:/',
                        help= 'Path to save the train and test data files.')
    return parser.parse_args()

def dataset_split_opt1(dataset, train_neg, test_neg): #testset with new users

    ratings = dataset.pivot(index = 'user_id', columns = 'item_id', values = 'rating')
    ratings.fillna(0, inplace=True)
    ratings=np.matrix(ratings)

    num_users, num_items = np.shape(ratings)
    #num_items = ratings.shape[1]
    users = dataset.user_id.unique()
    train_users, test_users = train_test_split(users, test_size = 0.2)
    nbr_train_users = len(train_users)
    
    #POSITIVES
        #test    
    testpos=[]
    indexes = dataset[dataset.user_id==-1].index
    for u in test_users:
        
        i = np.random.randint(num_items)
        while(ratings[u,i]==0):
            i = np.random.randint(num_items)
        testpos.append([[u,i,1]])
        indexes = indexes.append(dataset[dataset.user_id==u].index)
        
    testpos = np.vstack(testpos)
    test=pn.DataFrame(data=testpos,columns=['user_id','item_id','rating'])
    
        #train
    dataset.drop(indexes , inplace=True)
    train = dataset
    
    #NEGATIVES
        #test
    testneg = []

    for u in test_users:
        for num_neg in range(test_neg):
            
            i = np.random.randint(num_items)
            while(ratings[u,i]>0):
                i = np.random.randint(num_items)
            ratings[u,i]=1
            testneg.append([[u,i,0]])

    testneg = pn.DataFrame(data=np.vstack(testneg),columns=['user_id','item_id','rating'])
    
    test = test.append(testneg)
    test = test.sample(frac=1).reset_index(drop=True)

        #train
    trainneg=[]
    
    for x in range(train.shape[0]):
        for num_neg in range(train_neg):
            i = np.random.randint(num_items)
            u = train_users[np.random.randint(nbr_train_users)]
            while(ratings[u,i]>0):
                i = np.random.randint(num_items)
                u = train_users[np.random.randint(nbr_train_users)]
            ratings[u,i]=1
            trainneg.append([[u,i,0]])
    if(trainneg):        
        trainneg = pn.DataFrame(data=np.vstack(trainneg),columns=['user_id','item_id','rating'])
        train = train.append(trainneg)
        
    train = train.sample(frac=1).reset_index(drop=True)
    print(len(test.loc[test.rating==1]))
    return train,test

def dataset_split_opt2(dataset, train_neg, test_neg): #normal testset

    ratings = dataset.pivot(index = 'user_id', columns = 'item_id', values = 'rating')
    ratings.fillna(0, inplace=True)
    ratings=np.matrix(ratings)

    num_users, num_items = np.shape(ratings)
    
    #POSITIVES
        #test    
    testpos=[]
    indexes = dataset[dataset.user_id==-1].index
    for u in range(num_users):
        i = np.random.randint(num_items)
        while(ratings[u,i]==0):
            i = np.random.randint(num_items)
        testpos.append([[u,i,1]])
        indexes = indexes.append(dataset[(dataset.user_id==u) & (dataset.item_id==i)].index)
    testpos = np.vstack(testpos)
    test=pn.DataFrame(data=testpos,columns=['user_id','item_id','rating'])

        #train
    dataset.drop(indexes , inplace=True)
    train = dataset
    
    #NEGATIVES
        #test
    testneg = []

    for u in range(num_users):
        for num_neg in range(test_neg):
            
            i = np.random.randint(num_items)
            while(ratings[u,i]>0):
                i = np.random.randint(num_items)
            ratings[u,i]=1            
            testneg.append([[u,i,0]])
    testneg = pn.DataFrame(data=np.vstack(testneg),columns=['user_id','item_id','rating'])

    test = test.append(testneg)
    test = test.sample(frac=1).reset_index(drop=True)

        #train
    trainneg=[]

    for x in range(train.shape[0]):
        for num_neg in range(train_neg):
            
            i = np.random.randint(num_items)
            u = np.random.randint(num_users)
            while(ratings[u,i]>0):
                i = np.random.randint(num_items)
            ratings[u,i]=1            
            trainneg.append([[u,i,0]])
    
    if(trainneg):        
        trainneg = pn.DataFrame(data=np.vstack(trainneg),columns=['user_id','item_id','rating'])
        train = train.append(trainneg)
    train = train.sample(frac=1).reset_index(drop=True)
    return train, test

def occupation(users) : #for ML 100k
    users.loc[users.occupation == 'administrator', 'occupation'] = 0
    users.loc[users.occupation == 'artist', 'occupation'] = 1
    users.loc[users.occupation == 'doctor', 'occupation'] = 2
    users.loc[users.occupation == 'educator', 'occupation'] = 3
    users.loc[users.occupation == 'engineer', 'occupation'] = 4
    users.loc[users.occupation == 'entertainment', 'occupation'] = 5
    users.loc[users.occupation == 'executive', 'occupation'] = 6
    users.loc[users.occupation == 'healthcare', 'occupation'] = 7
    users.loc[users.occupation == 'homemaker', 'occupation'] = 8
    users.loc[users.occupation == 'lawyer', 'occupation'] = 9
    users.loc[users.occupation == 'librarian', 'occupation'] = 10
    users.loc[users.occupation == 'marketing', 'occupation'] = 11
    users.loc[users.occupation == 'none', 'occupation'] = 12
    users.loc[users.occupation == 'other', 'occupation'] = 13
    users.loc[users.occupation == 'programmer', 'occupation'] = 14
    users.loc[users.occupation == 'retired', 'occupation'] = 15
    users.loc[users.occupation == 'salesman', 'occupation'] = 16
    users.loc[users.occupation == 'scientist', 'occupation'] = 17
    users.loc[users.occupation == 'student', 'occupation'] = 18
    users.loc[users.occupation == 'technician', 'occupation'] = 19
    users.loc[users.occupation == 'writer', 'occupation'] = 20 
    return users

def genres_items(items):  #for ML 1m  
    result = []
    for index, row in items.iterrows():
            genre = row['genre']
            vec = np.zeros(19).astype(int)
            vec[0]=row['item_id']
            if 'Action' in genre:
                vec[1]=1
            if 'Adventure' in genre:
                vec[2]=1
                
            if 'Animation' in genre:
                vec[3]=1
                
            if 'Children\'s' in genre:
                vec[4]=1
                
            if 'Comedy' in genre:
                vec[5]=1
                
            if 'Crime' in genre:
                vec[6]=1
                
            if 'Documentary' in genre:
                vec[7]=1
                
            if 'Drama' in genre:
                vec[8]=1
                
            if 'Fantasy' in genre:
                vec[9]=1
                
            if 'Film-Noir' in genre:
                vec[10]=1
                
            if 'Horror' in genre:
                vec[11]=1
                
            if 'Musical' in genre:
                vec[12]=1
                
            if 'Mystery' in genre:
                vec[13]=1
                
            if 'Romance' in genre:
                vec[14]=1
                
            if 'Sci-Fi' in genre:
                vec[15]=1
                
            if 'Thriller' in genre:
                vec[16]=1
                
            if 'War' in genre:
                vec[17]=1
                
            if 'Western' in genre:
                vec[18]=1

            result.append(vec)
    result=np.vstack(result)

    new_items = pn.DataFrame(data=result, columns = ['item_id','Action' ,'Adventure','Animation','Children\'s','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'],)
    return new_items

def read_data(dataset, users_path, items_path, ratings_path) :
    #read ML 100k
    if dataset == 1 :
        users = pn.read_csv(users_path, delimiter = '|', encoding ='cp1252',
        names = ['user_id', 'sexe', 'age', 'occupation', 'zip'], engine = 'python')
        users = users.drop('zip', axis = 1)

        users['age'], users['sexe'] = users['sexe'], users['age'] #invert columns in ml100k

        users.loc[users.sexe == 'F', 'sexe'] = 0
        users.loc[users.sexe == 'M', 'sexe'] = 1

        users = occupation(users)

        items = pn.read_csv(items_path, delimiter = '|', encoding ='cp1252',
            names = ['item_id', 'title', 'release_date', 'video_release_date',
                    'IMDb_URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
                    'Children\'s' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
                    'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
                    'Thriller' , 'War' , 'Western'], engine = 'python')
        items = items.drop('title', axis = 1)
        items = items.drop('release_date', axis = 1) 
        items = items.drop('video_release_date', axis = 1) 
        items = items.drop('IMDb_URL' , axis = 1) 
        items = items.drop('unknown', axis = 1) 
        
        ratings = pn.read_csv(ratings_path, delimiter = ' |\t', encoding ='cp1252',
            names = ['user_id', 'item_id', 'rating', 'timestamp'], engine = 'python')
        ratings.drop('timestamp', axis=1, inplace=True)
        ratings['rating']=1
        
        users.user_id = users.user_id.astype('category').cat.codes.values
        items.item_id = items.item_id.astype('category').cat.codes.values

        ratings.user_id = ratings.user_id.astype('category').cat.codes.values
        ratings.item_id = ratings.item_id.astype('category').cat.codes.values

    #read ML 1m
    elif dataset == 2 :
        users = pn.read_csv(users_path, delimiter = '::',encoding ='cp1252',
            names = ['user_id','sexe', 'age','occupation','postal_code'], engine = 'python')
        users.drop('postal_code', axis=1, inplace=True)
        users.loc[users.sexe == 'F','sexe'] = 0
        users.loc[users.sexe == 'M','sexe'] = 1

        items = pn.read_csv(items_path, delimiter = '::',encoding ='cp1252',
            names = ['item_id','title','genre'], engine = 'python')
        items.drop('title', axis=1, inplace=True)

        ratings = pn.read_csv(ratings_path, delimiter = '::',encoding ='cp1252',
            names = ['user_id', 'item_id', 'rating', 'timestamp'], engine = 'python')
        ratings.drop('timestamp', axis=1, inplace=True)
        ratings['rating']=1

        ratings = pn.merge(ratings, users, left_on='user_id', right_on='user_id', how='left')
        ratings = pn.merge(ratings, items, left_on='item_id', right_on='item_id', how='left')

        ratings.user_id = ratings.user_id.astype('category').cat.codes.values
        ratings.item_id = ratings.item_id.astype('category').cat.codes.values
        
        items = ratings.loc[:,['item_id','genre']]
        items = items.drop_duplicates()
        items = genres_items(items)

        users = ratings.loc[:,['user_id','sexe', 'age','occupation']]
        users = users.drop_duplicates()

        ratings = ratings.iloc[:,0:3]

    #read Yelp
    else :
        columns_items = ['item_id','latitude','longitude','Breakfast & Brunch', 'American (Traditional)', 'Burgers', 'Fast Food', 
        'American (New)', 'Chinese', 'Pizza', 'Italian', 'Sandwiches', 'Sushi Bars', 'Japanese', 
        'Indian', 'Mexican', 'Vietnamese', 'Thai', 'Asian Fusion','Take-out', 'Wi-Fi', 
        'dessert','latenight','lunch','dinner','breakfast','brunch', 'Caters', 
        'Noise Level','Takes Reservations', 'Delivery', 'romantic','intimate','touristy',
        'hipster','divey','classy','trendy','upscale','casual',
        'Parking', 'Has TV','Outdoor Seating', 'Attire', 'Alcohol', 'Waiter Service', 
        'Accepts Credit Cards', 'Good for Kids', 'Good For Groups', 
        'Price Range', 'Wheelchair Accessible']
        
        columns_users = ['user_id','fans','average_stars','friends','vote_funny','useful','vote_cool','hot','more',
        'profile','cute','list','note','plain',
        'cool','funny','writer','photos']

        columns_reviews = ['user_id','item_id','rating']

        users = pn.read_csv(users_path, delimiter = '::',encoding ='cp1252',
            names = columns_users,engine = 'python')
        items = pn.read_csv(items_path, delimiter = '::',encoding ='cp1252',
            names = columns_items,engine = 'python')
        reviews = pn.read_csv(ratings_path, delimiter = '::',encoding ='cp1252',
            names =columns_reviews , engine = 'python')

        ratings = reviews.loc[:,columns_reviews]
        
    return users, items, ratings

def load_data(dataset, users_path, items_path, ratings_path, save, num_neg, option) :
    #read the data
    users, items, ratings = read_data(dataset, users_path, items_path, ratings_path)

    #options of splitting the dataset
    if option == 1 :
        train, test = dataset_split_opt1(ratings, num_neg, 99)
    elif option == 2 :
        train, test = dataset_split_opt2(ratings, num_neg, 99)

    train = pn.merge(train, users, left_on='user_id', right_on='user_id', how='left')
    train = pn.merge(train, items, left_on='item_id', right_on='item_id', how='left')

    test = pn.merge(test,users, left_on='user_id', right_on='user_id', how='left')
    test = pn.merge(test,items, left_on='item_id', right_on='item_id', how='left')

    np.savetxt(save+'train.csv', train, delimiter='::', fmt='%s')
    np.savetxt(save+'test.csv', test, delimiter='::', fmt='%s')

if __name__ == '__main__' :
    args = parse_args()
    saving_path = args.path_save
    dataset = args.dataset
    num_neg = args.negs
    option = args.type
    
    path_users, path_items, path_ratings = '', '', ''
    
    #PATHS TO CHANGE
    if dataset == 1 :
        #ML 100k
        path_users = 'D:/Datasets/ml-100k/u.user'
        path_items = 'D:/Datasets/ml-100k/u.item'
        path_ratings = 'D:/Datasets/ml-100k/u.data'
    elif dataset == 2 :
        #ML 1m
        path_users = 'D:/Datasets/ml-1m/users.dat'
        path_items = 'D:/Datasets/ml-1m/movies.dat'
        path_ratings = 'D:/Datasets/ml-1m/ratings.dat'
    else :
        #Yelp
        path_users = 'D:/Yelp-data/users.csv'
        path_items = 'D:/Yelp-data/restaurants.csv'
        path_ratings = 'D:/Yelp-data/reviews.csv'

    print('Loading & Processing data...')
    load_data(dataset, path_users, path_items, path_ratings, saving_path, num_neg, option)
    print('Data loaded and processed.')