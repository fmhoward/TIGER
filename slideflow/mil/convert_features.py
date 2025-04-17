import slideflow as sf
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from slideflow.mil.model_AE import *

#Performs KMeans clustering as per Sequoia
def convert_to_sequoia(bags, outdir, n_clusters = 100):
    paths = [f for f in os.listdir(bags) if f.endswith('pt')]
    for i in paths:
        slide = sf.util.path_to_name(i)
        features = torch.load(os.path.join(bags, f'{slide}.pt'))
        if len(features) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
            clusters = kmeans.labels_
            mean_features = []
            for pos in range(n_clusters):
                indexes = np.where(clusters == pos)
                features_aux = features[indexes]
                if features_aux.shape[0] > 0:  # Ensure it's not empty
                    mean_features.append(features_aux.cpu().numpy().mean(axis=0))
                else:
                    mean_features.append(np.zeros(features.shape[1]))  # Placeholder for empty clusters
            mean_features = np.asarray(mean_features)
            tensor_data = torch.tensor(mean_features) 
            torch.save(tensor_data, os.path.join(outdir, f'{slide}.pt'))
            
#Trains an autoencoder for feature extraction as per DeepPT
def convert_to_deeppt(bags, outdir, model_path = None, max_epochs = 500, n_inputs = 1024, n_hiddens = 512, n_outputs = 1024, batch_size = 2048, lr = 0.0001):
    
    paths = [f for f in os.listdir(bags) if f.endswith('pt')]
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    if model_path is None:
        features_list = []  # Store tensors in a list for efficient concatenation
        for i in paths:
            slide = sf.util.path_to_name(i)
            features1 = torch.load(os.path.join(bags, f'{slide}.pt'))
            features_list.append(features1)  # Append to list instead of concatenating immediately

        features = torch.cat(features_list) 
        model = AutoEncoder(n_inputs, n_hiddens, n_outputs)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_idx, test_idx = train_test_split(features, test_size=0.1, shuffle=True)
        model,train_loss,test_loss = fit(model, optimizer, train_idx, test_idx, max_epochs, batch_size, device)
        
        torch.save(model.state_dict(), os.join(outdir, "model_AE.pth"))
    else:
        model = AutoEncoder(n_inputs, n_hiddens, n_outputs)
        model.to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
    for i in paths:        
        slide = sf.util.path_to_name(i)
        features_AE1 = features_compression(model,bags,i,device)
        torch.save(features_AE1, os.path.join(outdir, f'{slide}.pt'))