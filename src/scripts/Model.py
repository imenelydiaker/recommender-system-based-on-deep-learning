import pandas as pn
import numpy as np
import math
from keras.layers.core import Flatten
from keras.layers import Embedding, Multiply, Dense, Input
from keras.models import Model
from keras.layers.merge import concatenate
from keras.models import load_model
from time import time
import pickle
import argparse
import matplotlib.pyplot as plt

def parse_args() :
    parser = argparse.ArgumentParser(description="Training the model.")
    parser.add_argument('--path_save', nargs = '?', default= 'D:/',
                        help= 'Path to save the results.')
    parser.add_argument('--model', type = int, default= 1,
                        help= 'Model to train : 1)GMF  2)HybMLP  3)NHybNeuMF 4)MLP  5)NCF.')
    parser.add_argument('--pretrain', type = int, default= 0,
                        help= 'Pretrain : Enable 1, Disable 0.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs.')
    parser.add_argument('--optimizer_subs', nargs='?', default='adam', 
                        help='Specify an optimizer for the submodels : adam, sgd.')
    parser.add_argument('--optimizer_nhf', nargs='?', default='sgd', 
                        help='Specify an optimizer for NHybF : adam, sgd.')
    parser.add_argument('--optimizer_ncf',  nargs='?' , default='sgd', 
                        help='Specify an optimizer for NCF : adam, sgd.')
    parser.add_argument('--batch_size', type=int, default=100, 
                        help ='Batch size.')
    parser.add_argument('--num_layers', type=int, default=5, 
                        help ='HybMLP number of layers.')
    parser.add_argument('--emb_size_gmf', type=int, default=64, 
                        help ='GMF embedding size.')
    parser.add_argument('--emb_size_hmlp', type=int, default=32, 
                        help ='HybMLP embedding size.')
    parser.add_argument('--emb_size_mlp', type=int, default=32, 
                        help ='MLP embedding size.')
    parser.add_argument('--num_factors', type=int, default=16,
                        help ='Number of predictive factors for HybMLP.')
    parser.add_argument('--num_neg', type=int, default=1, 
                        help ='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--estop', type = int, default = -1,
                        help = 'Early Stopping : Enable 1, Disable -1.')
    parser.add_argument('--path_data', nargs = '?', default= 'D:/',
                        help = 'Input trainset and testset folder path.')
    parser.add_argument('--dataset', type = int, default = 1,
                        help = 'Choose a dataset:\n 1)MovieLens 100K  2)MovieLens 1M  3)Yelp.')
    return parser.parse_args()

def save_obj(obj, path):
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_data(path, dataset):
    if dataset == 3 :
        yelp_columns = ['user_id','item_id', 'rating', 'fans','average_stars','friends','vote_funny','useful','vote_cool','hot','more',
        'profile','cute','list','note','plain', 'cool','funny','writer','photos', 
        'latitude','longitude','Breakfast & Brunch', 'American (Traditional)', 'Burgers', 'Fast Food', 
        'American (New)', 'Chinese', 'Pizza', 'Italian', 'Sandwiches', 'Sushi Bars', 'Japanese', 
        'Indian', 'Mexican', 'Vietnamese', 'Thai', 'Asian Fusion','Take-out', 'Wi-Fi', 
        'dessert','latenight','lunch','dinner','breakfast','brunch', 'Caters', 
        'Noise Level','Takes Reservations', 'Delivery', 'romantic','intimate','touristy',
        'hipster','divey','classy','trendy','upscale','casual',
        'Parking', 'Has TV','Outdoor Seating', 'Attire', 'Alcohol', 'Waiter Service', 
        'Accepts Credit Cards', 'Good for Kids', 'Good For Groups', 
        'Price Range', 'Wheelchair Accessible']

        trainset = pn.read_csv(path+'train.csv', delimiter = '::',
        names = yelp_columns, engine= 'python')

        testset = pn.read_csv(path+'test.csv', delimiter = '::',
        names = yelp_columns, engine= 'python')
    else :
        movielens_columns = ['user_id', 'item_id', 'rating','sexe', 'age','occupation', 'Action' ,
        'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        trainset = pn.read_csv(path+'train.csv', delimiter = '::',
        names = movielens_columns, engine = 'python')
        
        testset = pn.read_csv(path+'test.csv', delimiter = '::',
        names = movielens_columns, engine = 'python')
        
    num_users = int(max(testset.user_id.max()+1,trainset.user_id.max()+1))
    num_items = int(max(testset.item_id.max()+1,trainset.item_id.max()+1))
    
    return trainset, testset, num_users, num_items

def evaluate_model(model, model_name, test, k, dataset):
    if dataset == 3 :
        #Yelp
        test_userID = test['user_id']
        test_userDATA = test[['fans','average_stars','friends','vote_funny','useful','vote_cool','hot','more',
        'profile','cute','list','note','plain', 'cool','funny','writer','photos']] 

        test_itemID = test['item_id']
        test_itemDATA = test.iloc[:,-50:]
    else :
        #MovieLens
        test_userID = test['user_id']
        test_userDATA = test[['sexe', 'age','occupation']] 

        test_itemID = test['item_id']
        test_itemDATA = test.iloc[:,-18:]

    if(model_name in ['GMF','NCF','MLP','pretrained NCF']):
        predictions = model.predict([test_userID, test_itemID], verbose=2)
    else:
        predictions = model.predict([test_userID, test_userDATA, test_itemID, test_itemDATA], verbose=2)
        
    predictions = pn.DataFrame(data=predictions,columns=['predicted'])
    predictions = pn.concat([test,predictions], axis=1)
    predictions = predictions.sort_values(by=['predicted'], ascending=False)
    
    users=test.user_id.unique()
    
    hrs, ndcgs = 0,0
    for u in users:
        p= predictions[predictions['user_id']==u].loc[:,'rating'].head(k)
        hr, ndcg = evaluate_by_user(p)
        hrs = hrs + hr
        ndcgs = ndcgs + ndcg
        
    mean_hr = hrs/len(users)
    mean_ndcg = ndcgs/len(users)
        
    return mean_hr, mean_ndcg

def evaluate_by_user(test):
    hr, ndcg = 0, 0
    i=1
    for rating in test:
        if rating == 1 :
            hr = 1
            ndcg = math.log(2)/math.log(i+1)
            break 
        i=i+1    
    return hr, ndcg

#create model 
def build_NHF_model(emb_size_hmlp, emb_size_gmf, predictive_factors, num_layers, dataset):
    input_userID = Input(shape = [1], name = 'user_ID')
    input_itemID = Input(shape = [1], name = 'item_ID')

    if dataset == 3 : 
        #Yelp
        input_userDATA = Input(shape = [17], name = 'user_data') 
        input_itemDATA = Input(shape = [50], name = 'item_data')
    else : 
        #MovieLens
        input_userDATA = Input(shape = [3], name = 'user_data') 
        input_itemDATA = Input(shape = [18], name = 'item_data') 

    #GMF part
    user_latent_factors_GMF = emb_size_gmf
    item_latent_factors_GMF = emb_size_gmf

    user_emb_GMF = Embedding(num_users, user_latent_factors_GMF, name = 'user_emb_GMF')(input_userID)
    item_emb_GMF = Embedding(num_items, item_latent_factors_GMF, name = 'item_emb_GMF')(input_itemID)

    flat_u_GMF = Flatten()(user_emb_GMF)
    flat_i_GMF = Flatten()(item_emb_GMF)

    mul_layer = Multiply()([flat_u_GMF, flat_i_GMF])

    #HybMLP part
    user_latent_factors_hMLP = emb_size_hmlp
    item_latent_factors_hMLP = emb_size_hmlp

    user_emb_hMLP = Embedding(num_users, user_latent_factors_hMLP, name= 'user_emb_hMLP')(input_userID)
    item_emb_hMLP = Embedding(num_items, item_latent_factors_hMLP, name= 'item_emb_hMLP')(input_itemID)

    flat_u_hMLP = Flatten()(user_emb_hMLP)
    flat_i_hMLP = Flatten()(item_emb_hMLP)

    concat_hMLP = concatenate([flat_u_hMLP, flat_i_hMLP, input_userDATA, input_itemDATA])
    layer = concat_hMLP
    for l in range(num_layers,0,-1):
        layer = Dense(predictive_factors*(2**(l-1)), activation='relu', name= 'layer%d' %(num_layers-l+1))(layer)
    
    #NeuHybMF part
    concat_NeuhMF = concatenate([mul_layer,layer])
    
    out = Dense(1, activation='sigmoid', name='output')(concat_NeuhMF)

    model = Model([input_userID, input_userDATA, input_itemID, input_itemDATA], out)
    
    return model

#create GMF model
def build_GMF_model(emb_size):
    input_userID = Input(shape = [1], name = 'user_ID')
    input_itemID = Input(shape = [1], name = 'item_ID')

    user_emb_GMF = Embedding(num_users, emb_size, name = 'user_emb_GMF')(input_userID)
    item_emb_GMF = Embedding(num_items, emb_size, name = 'item_emb_GMF')(input_itemID)

    flat_u_GMF = Flatten()(user_emb_GMF)
    flat_i_GMF = Flatten()(item_emb_GMF)

    mul_layer = Multiply()([flat_u_GMF, flat_i_GMF])
    
    out = Dense(1, activation='sigmoid', name='output')(mul_layer)

    GMF_model = Model([input_userID, input_itemID], out)

    return GMF_model

#create HybMLP model
def build_hMLP_model(emb_size, predictive_factors, num_layers, dataset):
    input_userID = Input(shape = [1], name = 'user_ID')
    input_itemID = Input(shape = [1], name = 'item_ID')

    if dataset == 3 : 
        #Yelp
        input_userDATA = Input(shape = [17], name = 'user_data') 
        input_itemDATA = Input(shape = [50], name = 'item_data')
    else : 
        #MovieLens
        input_userDATA = Input(shape = [3], name = 'user_data') 
        input_itemDATA = Input(shape = [18], name = 'item_data') 

    user_emb_hMLP = Embedding(num_users, emb_size, name= 'user_emb_hMLP')(input_userID)
    item_emb_hMLP = Embedding(num_items, emb_size, name= 'item_emb_hMLP')(input_itemID)

    flat_u_hMLP = Flatten()(user_emb_hMLP)
    flat_i_hMLP = Flatten()(item_emb_hMLP)
    
    concat_hMLP = concatenate([flat_u_hMLP, flat_i_hMLP, input_userDATA, input_itemDATA])
    layer = concat_hMLP
    for l in range(num_layers,0,-1):
        layer = Dense(predictive_factors*(2**(l-1)), activation='relu', name= 'layer%d' %(num_layers-l+1))(layer)
        
    out = Dense(1, activation='sigmoid', name='output')(layer)

    hMLP_model = Model([input_userID, input_userDATA, input_itemID, input_itemDATA], out)

    return hMLP_model

#create NCF model
def build_NCF_model(emb_size_mlp, emb_size_gmf, predictive_factors, num_layers):
    input_userID = Input(shape = [1], name = 'user_ID')
    input_itemID = Input(shape = [1], name = 'item_ID')

    #GMF part
    user_latent_factors_GMF = emb_size_gmf
    item_latent_factors_GMF = emb_size_gmf

    user_emb_GMF = Embedding(num_users, user_latent_factors_GMF, name = 'user_emb_GMF')(input_userID)
    item_emb_GMF = Embedding(num_items, item_latent_factors_GMF, name = 'item_emb_GMF')(input_itemID)

    flat_u_GMF = Flatten()(user_emb_GMF)
    flat_i_GMF = Flatten()(item_emb_GMF)

    mul_layer = Multiply()([flat_u_GMF, flat_i_GMF])

    #MLP part
    user_latent_factors_MLP = emb_size_mlp
    item_latent_factors_MLP = emb_size_mlp

    user_emb_MLP = Embedding(num_users, user_latent_factors_MLP, name= 'user_emb_MLP')(input_userID)
    item_emb_MLP = Embedding(num_items, item_latent_factors_MLP, name= 'item_emb_MLP')(input_itemID)

    flat_u_MLP = Flatten()(user_emb_MLP)
    flat_i_MLP = Flatten()(item_emb_MLP)

    concat_MLP = concatenate([flat_u_MLP, flat_i_MLP])
    layer = concat_MLP
    for l in range(num_layers,0,-1):
        layer = Dense(predictive_factors*(2**(l-1)), activation='relu', name= 'layer%d' %(num_layers-l+1))(layer)
    
    #NeuMF part
    concat_NeuhMF = concatenate([mul_layer,layer])
    
    out = Dense(1, activation='sigmoid', name='output')(concat_NeuhMF)

    model = Model([input_userID, input_itemID], out)
    
    return model


#create MLP model
def build_MLP_model(emb_size,predictive_factors,num_layers):
    input_userID = Input(shape = [1], name = 'user_ID')
    input_itemID = Input(shape = [1], name = 'item_ID')
    
    user_latent_factors_MLP = emb_size
    item_latent_factors_MLP = emb_size

    user_emb_MLP = Embedding(num_users, user_latent_factors_MLP, name= 'user_emb_MLP')(input_userID)
    item_emb_MLP = Embedding(num_items, item_latent_factors_MLP, name= 'item_emb_MLP')(input_itemID)

    flat_u_MLP = Flatten()(user_emb_MLP)
    flat_i_MLP = Flatten()(item_emb_MLP)
    
    concat_MLP = concatenate([flat_u_MLP, flat_i_MLP])
    layer = concat_MLP
    for l in range(num_layers,0,-1):
        layer = Dense(predictive_factors*(2**(l-1)), activation='relu', name= 'layer%d' %(num_layers-l+1))(layer)
        
    out = Dense(1, activation='sigmoid', name='output')(layer)

    MLP_model = Model([input_userID, input_itemID], out)

    return MLP_model

def load_pretrained_model(model, model_name, gmf_model, hmlp_model, num_layers):
    # MF embeddings
    user_emb_GMF = gmf_model.get_layer('user_emb_GMF').get_weights()
    item_emb_GMF = gmf_model.get_layer('item_emb_GMF').get_weights()
    model.get_layer('user_emb_GMF').set_weights(user_emb_GMF)
    model.get_layer('item_emb_GMF').set_weights(item_emb_GMF)
    if(model_name=='NHF'):
        # HybMLP embeddings
        user_emb_MLP = hmlp_model.get_layer('user_emb_hMLP').get_weights()
        item_emb_MLP = hmlp_model.get_layer('item_emb_hMLP').get_weights()
        model.get_layer('user_emb_hMLP').set_weights(user_emb_MLP)
        model.get_layer('item_emb_hMLP').set_weights(item_emb_MLP)
    elif(model_name=='NCF'):
        # MLP embeddings
        user_emb_MLP = hmlp_model.get_layer('user_emb_MLP').get_weights()
        item_emb_MLP = hmlp_model.get_layer('item_emb_MLP').get_weights()
        model.get_layer('user_emb_MLP').set_weights(user_emb_MLP)
        model.get_layer('item_emb_MLP').set_weights(item_emb_MLP)
    # HybMLP/MLP layers
    for i in range(num_layers):
        mlp_layer_weights = hmlp_model.get_layer('layer%d' %(i+1)).get_weights()
        model.get_layer('layer%d' %(i+1)).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('output').get_weights()
    mlp_prediction = hmlp_model.get_layer('output').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('output').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

def train(model, model_name, train, test, num_epochs, path, batch, dataset):
    if dataset == 3 :
        #Yelp
        train_userID = train['user_id']
        train_userDATA = train[['fans','average_stars','friends','vote_funny','useful','vote_cool','hot','more',
        'profile','cute','list','note','plain','cool','funny','writer','photos']] 
        

        train_itemID = train['item_id']
        train_itemDATA = train.iloc[:,-50:] 
        
        train_y = train['rating']
    else : 
        #MovieLens
        train_userID = train['user_id']
        train_userDATA = train[['sexe', 'age','occupation']] 
        
        train_itemID = train['item_id']
        train_itemDATA = train.iloc[:,-18:] 
        
        train_y = train['rating']

    print('\nTRAINING '+model_name+'...\n')

    #intitialisation        
    best_hr, best_ndcg = evaluate_model(model,model_name,test, 10, dataset)
    best_iteration = 0
    all_hrs, all_ndcgs = {}, {}
    all_hrs[0], all_ndcgs[0] = best_hr, best_ndcg
    model.save_weights(path+' '+model_name+'.h5', overwrite = True)
    bad_epochs=0
    
    #for every epoch : train and test
    for epoch in range(1, num_epochs+1) :
        t1 = time()
        
        if(model_name in ['GMF','NCF','MLP','pretrained NCF']):
            history = model.fit([train_userID, train_itemID], train_y, batch_size = batch , epochs = 1, verbose = 0, shuffle = True)
        else:
            history = model.fit([train_userID, train_userDATA, train_itemID, train_itemDATA], train_y, batch_size = batch , epochs = 1, verbose = 0, shuffle = True)
        
        hr, ndcg = evaluate_model(model, model_name, test, 10, dataset)
        all_hrs[epoch], all_ndcgs[epoch] = hr, ndcg
        
        if ndcg > best_ndcg :
            best_hr, best_ndcg, best_iteration = hr, ndcg, epoch
            model.save_weights(path+' '+model_name+'.h5', overwrite = True)
            bad_epochs = 0
        else:
            bad_epochs=bad_epochs+1
            if(bad_epochs==early_stopping):
                break
        
        t2 = time()
        print('Iteration %d [%.2f s]: loss = %.4f, HR = %.4f, NDCG = %.4f' % (epoch, t2-t1, history.history['loss'][0], hr, ndcg))
               
    print("Best iteration %d, best HR = %.4f, best NDCG = %.4f" % (best_iteration, best_hr, best_ndcg))
    save_obj(all_hrs,path+' '+model_name+' HRs')
    save_obj(all_ndcgs,path+' '+model_name+' NDCGs')
    return all_hrs, all_ndcgs

if __name__ == "__main__" :

    args = parse_args()
    path_save = args.path_save
    dataset = args.dataset
    model_to_train = int(args.model)
    pretrain = int(args.pretrain)
    epochs = args.epochs
    optimizer_nhf = args.optimizer_nhf.lower()
    optimizer_subs = args.optimizer_subs.lower()
    optimizer_ncf = args.optimizer_ncf.lower()
    emb_size_gmf = int(args.emb_size_gmf)
    emb_size_hmlp = int(args.emb_size_hmlp)
    emb_size_mlp = args.emb_size_mlp #check cast
    pred_facts = int(args.num_factors)
    num_neg = int(args.num_neg)
    num_layers = int(args.num_layers)
    early_stopping = int(args.estop)
    path_data = args.path_data
    batch = int(args.batch_size)

    '''
    number of epochs (early stopping)
    '''

    #load data
    trainset, testset, num_users, num_items = load_data(path_data, dataset)
    
    #select model to train
    if model_to_train == 1 :
        #train GMF
        path = path_save+'EMB %d'%(emb_size_gmf)
        GMF = build_GMF_model(emb_size_gmf)
        GMF.compile(optimizer = optimizer_subs, loss = 'binary_crossentropy')
        hrs,ndcgs = train(GMF, 'GMF', trainset, testset, epochs, path, batch, dataset)

    elif model_to_train == 2 :
        #train HybMLP
        path = path_save+'HybMLP %d EMB %d PF %d'%(num_layers, emb_size_hmlp, pred_facts)
        hMLP = build_hMLP_model(emb_size_hmlp, pred_facts, num_layers, dataset)
        hMLP.compile(optimizer = optimizer_subs, loss = 'binary_crossentropy')
        hrs,ndcgs = train(hMLP, 'HybMLP', trainset, testset, epochs, path, batch, dataset)

    elif  model_to_train == 3 :
        #train NeuHybMF
        if pretrain == 1 : #with pretrain models
            #pretrain GMF
            path = path_save+'EMB %d'%(emb_size_gmf)
            GMF = build_GMF_model(emb_size_gmf)
            GMF.compile(optimizer = optimizer_subs, loss = 'binary_crossentropy')
            train(GMF, 'GMF', trainset, testset, epochs, path, batch, dataset)

            #pretrain HybMLP
            
            if num_layers!=0:
                path = path_save+'HybMLP %d EMB %d PF %d'%(num_layers, emb_size_hmlp, pred_facts)
            else:
                path = path_save+'HybMLP %d EMB %d'%(num_layers, emb_size_hmlp)

            hMLP = build_hMLP_model(emb_size_hmlp, pred_facts, num_layers, dataset)
            hMLP.compile(optimizer = optimizer_subs, loss = 'binary_crossentropy')
            train(hMLP, 'HybMLP', trainset, testset, epochs, path, batch, dataset)

            #load pretrained models and train NeuHybMF (NHF)
            NHF = build_NHF_model(emb_size_hmlp, emb_size_gmf, pred_facts, num_layers, dataset)
            NHF.compile(optimizer = optimizer_nhf, loss = 'binary_crossentropy')
            NHF = load_pretrained_model(NHF,'NHF', GMF, hMLP, num_layers)
            hrs,ndcgs = train(NHF, 'pretrained NHF', trainset, testset, epochs, path, batch, dataset)

        else : #without pretrain models
            NHF = build_NHF_model(emb_size_hmlp, emb_size_gmf, pred_facts, num_layers, dataset)
            NHF.compile(optimizer = optimizer_nhf, loss = 'binary_crossentropy')
            hrs,ndcgs = train(NHF,'NHF', trainset, testset, epochs, path_save, batch, dataset)

    elif model_to_train == 4 :
        #train MLP
        if num_layers!=0:
            path = path_save+'HybMLP %d EMB %d PF %d'%(num_layers, emb_size_hmlp, pred_facts)
        else:
            path = path_save+'HybMLP %d EMB %d'%(num_layers, emb_size_hmlp)
        
        MLP = build_MLP_model(emb_size_mlp, pred_facts, num_layers)
        MLP.compile(optimizer = optimizer_subs, loss = 'binary_crossentropy')
        hrs,ndcgs = train(MLP, 'MLP', trainset, testset, epochs, path, batch, dataset)

    elif  model_to_train == 5 :
        #train NCF
        if pretrain == 1 : #with pretrain models
            #pretrain GMF
            path = path_save+'EMB %d'%(emb_size_gmf)
            GMF = build_GMF_model(emb_size_gmf)
            GMF.compile(optimizer = optimizer_subs, loss = 'binary_crossentropy')
            train(GMF, 'GMF', trainset, testset, epochs, path, batch, dataset)

            #train MLP
            if num_layers!=0:
                path = path_save+'HybMLP %d EMB %d PF %d'%(num_layers, emb_size_hmlp, pred_facts)
            else:
                path = path_save+'HybMLP %d EMB %d'%(num_layers, emb_size_hmlp)
                
            MLP = build_MLP_model(emb_size_mlp, pred_facts, num_layers)
            MLP.compile(optimizer = optimizer_subs, loss = 'binary_crossentropy')
            train(MLP, 'MLP', trainset, testset, epochs, path, batch, dataset)

            #load pretrained models and train NCF
            NCF = build_NCF_model(emb_size_mlp, emb_size_gmf, pred_facts, num_layers)
            NCF.compile(optimizer = optimizer_ncf, loss = 'binary_crossentropy')
            NCF = load_pretrained_model(NCF,'NCF', GMF, MLP, num_layers)
            hrs,ndcgs = train(NCF, 'pretrained NCF', trainset, testset, epochs, path, batch, dataset)

        else : #without pretrain models
            NCF = build_NCF_model(emb_size_mlp, emb_size_gmf, pred_facts, num_layers)
            NCF.compile(optimizer = optimizer_ncf, loss = 'binary_crossentropy')
            hrs,ndcgs = train(NCF,'NCF', trainset, testset, epochs, path_save, batch, dataset)

    #PLOT
    
    x1, y1 = zip(*(hrs.items()))
    x2, y2 = zip(*(ndcgs.items()))
    plt.plot(x1,y1,'ro',linestyle = 'dashed')

    plt.subplot(211)
    plt.plot(x1,y1,'ro',linestyle = 'dashed')
    plt.title('The results of the training')
    plt.ylabel('HR@10')   
    plt.subplot(212)
    plt.plot(x2,y2,'bd',linestyle = 'dashed')
    plt.ylabel('NDCG@10')
    plt.xlabel('Epochs')
    plt.show() 
