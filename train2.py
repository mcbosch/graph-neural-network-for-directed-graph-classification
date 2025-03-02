'''
Aquest programa és una versió simplificada
del programa train.py per treballar únicament
amb els següents models:
 > GCNN
 > SGCNN
 > MAGNET
'''

'''
References: 
'''
import argparse
from argparse import RawTextHelpFormatter
import sys

# Carregam els paquets per calcular els resultats
import numpy as np
import time
import statistics


# Carregam els paquets necessaris per manipulars els grafs i les GNNs
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary

# Carregam les dades e importam els models que hem definit
from datasets.graph_data_reader import DataReader, GraphData
from models.GCN import GCN
from models.SGCNN import SGCNN

# Codi per guardar resultats
from utils import create_directory, save_result_csv

model_list = ['GCN', 'SGCN', 'MAGNET']
dataset_list = ['PROTEINS']
readout_list = ['max', 'avg', 'sum']

#====================================================================================
#           PARÀMETRES DEL PROGRAMA 
#====================================================================================
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

# Target model & dataset & readout param
parser.add_argument('--model_list', nargs='+', required=True,
                    help='target train model list \n'+
                    'available model: ' + str(model_list)
                     + '\nable to choose multiple models \n'+
                     'ALL: choose all available model')
parser.add_argument('--dataset_list', nargs='+', required=True,
                    help='target graph classification dataset list \n'+
                    'available dataset: ' + str(dataset_list)
                     + '\nable to choose multiple datasets \n'+
                     'ALL: choose all available dataset')
parser.add_argument('--readout_list', nargs='+', required=True,
                    help='target readout method list \n'+
                    'available readout: ' + str(readout_list)
                     + '\nable to choose multiple readout methods \n'+
                     'ALL: choose all available readout')
                     
# Dataset param
parser.add_argument('--node_att', default='FALSE',
                    help='use additional float valued node attributes available in some datasets or not\n'+
                    'TRUE/FALSE')
parser.add_argument('--seed', type=int, default=111,
                    help='random seed')
parser.add_argument('--n_folds', type=int, default=10,
                    help='the number of folds in 10-cross validation')
parser.add_argument('--threads', type=int, default=0,
                    help='how many subprocesses to use for data loading \n'+
                    'default value 0 means that the data will be loaded in the main process')

# Graph data subsampling
parser.add_argument('--n_graph_subsampling', type=int, default=0,
                    help='the number of running graph subsampling each train graph data\n'+
                    'run subsampling 5 times: increasing graph data 5 times')
parser.add_argument('--graph_node_subsampling', default='TRUE',
                    help='TRUE: removing node randomly to subsampling and augmentation of graph dataset \n'+
                    'FALSE: removing edge randomly to subsampling and augmentation of graph dataset')
parser.add_argument('--graph_subsampling_rate', type=float, default=0.2,
                    help='graph subsampling rate')

# Learning param
parser.add_argument('--cuda', default='TRUE',
                    help='use cuda device in train process or not\n'+
                    'TRUE/FALSE')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size of data')                  
parser.add_argument('--epochs', type=int, default=50,
                    help='train epochs')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate of optimizer')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay of optimizer')
parser.add_argument('--log_interval', type=int, default=10,
                    help='print log interval in train process')
parser.add_argument('--save_model', default='TRUE',
                    help='save model or not\n'+
                    'TRUE/FALSE')

# Model param
parser.add_argument('--n_agg_layer', type=int, default=2,
                    help='the number of graph aggregation layers')
parser.add_argument('--agg_hidden', type=int, default=64,
                    help='size of hidden graph aggregation layer')
parser.add_argument('--fc_hidden', type=int, default=128,
                    help='size of fully-connected layer after readout')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout rate of layer')

#=================================================================================
#          ENTRENAMENT      
#=================================================================================

# Lletgim els arguments que ha introduit l'usuari
args = parser.parse_args()

# Quin device farem servir
args.cuda = (args.cuda.upper()=='TRUE') 
# Guardam els models o no; En cas de que si on es guarden?
args.save_model = (args.save_model.upper()=='TRUE') 
# Hi ha o no atributs en els nodes
args.node_att = (args.node_att.upper()=='TRUE') 
# Hem de prendre un subconjunt de nodes; que es això??¿?
args.graph_node_subsampling = (args.graph_node_subsampling.upper()=='TRUE')

# Choose target graph classification model
if 'ALL' in args.model_list:
  args.model_list = model_list
else:
  for model in args.model_list:
    if not model in model_list:
      print('There are not available models in the target graph classification model list')
      print('The models avaibles are:', model_list)
      sys.exit()

print('Target model list:', args.model_list)

# Choose target dataset
if 'ALL' in args.dataset_list:
  args.dataset_list = dataset_list
else:
  for dataset in args.dataset_list:
    if not dataset in dataset_list:
      print('There are not available datasets in the target graph dataset list')
      print('The datasets avaibles are:', dataset_list)
      sys.exit()

print('Target dataset list:', args.dataset_list)

# Choose target readout
if 'ALL' in args.readout_list:
  args.readout_list = readout_list
else:
  for readout in args.readout_list:
    if not readout in readout_list:
      print('There are not available readouts in the target readout list')
      print('The readouts avaible are:', readout_list)
      sys.exit()

print('Target readout list:', args.readout_list)

# Choose device
if args.cuda and torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
  
print('Using device in train process:', device)

#-------------------------------------- Fins aquí okey
#   Coses pendents a entendre:
#       o Què són els threads
#       o Recompte dels paràmetres per entrenar
#       o Funció entrenament

if args.n_graph_subsampling > 0 and args.graph_node_subsampling:
  print('graph subsampling: random node removal')
elif args.n_graph_subsampling > 0 and not args.graph_node_subsampling:
  print('graph subsampling: random edge removal')

for dataset_name in args.dataset_list:
    print('-'*50)
    
    print('Target dataset:', dataset_name)

    # Build graph data reader: IMDB-BINARY, IMDB-MULTI, ...
    datareader = DataReader(data_dir='./datasets/%s/' % dataset_name.upper(),
                        rnd_state=np.random.RandomState(args.seed),
                        folds=args.n_folds,           
                        use_cont_node_attr=False,
                        random_walk=None,
                        num_walk=args.num_walk,
                        walk_length=args.walk_length,
                        p=args.p,
                        q=args.q,
                        node2vec_hidden=args.agg_hidden
                        )
    # Prenem un model
    for model_name in args.model_list:
      
      # Prenem una funció readout
      for i, readout_name in enumerate(args.readout_list):
        print('-'*25)
        
        # Build graph classification model
        if model_name == 'GCN':
            model = GCN(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'SGCNN':
            model = SGCNN(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)                                                 
        print(model)
        print('Readout:', readout_name)
        
        # Train & test each fold
        acc_folds = []
        time_folds = []
        for fold_id in range(args.n_folds):
            print('\nFOLD', fold_id)
            loaders = []
            for split in ['train', 'test']:
                # Build GDATA object
                if split == 'train':
                    gdata = GraphData(fold_id=fold_id,
                                       datareader=datareader,
                                       split=split,
                                       random_walk=random_walk,
                                       n_graph_subsampling=args.n_graph_subsampling,
                                       graph_node_subsampling=args.graph_node_subsampling,
                                       graph_subsampling_rate=args.graph_subsampling_rate)
                else:
                    gdata = GraphData(fold_id=fold_id,
                                       datareader=datareader,
                                       split=split,
                                       random_walk=random_walk,
                                       n_graph_subsampling=0,
                                       graph_node_subsampling=args.graph_node_subsampling,
                                       graph_subsampling_rate=args.graph_subsampling_rate)      
                
                # Build graph data pytorch loader
                loader = torch.utils.data.DataLoader(gdata, 
                                                     batch_size=args.batch_size,
                                                     shuffle=split.find('train') >= 0,
                                                     num_workers=args.threads,
                                                     drop_last=False)
                loaders.append(loader)
            
            # Total trainable param
            c = 0
            for p in filter(lambda p: p.requires_grad, model.parameters()):
                c += p.numel()
            print('N trainable parameters:', c)
            
            # Optimizer
            optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        betas=(0.5, 0.999))
    
            scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.1)
            
            # Train function
            def train(train_loader):
                total_time_iter = 0
                model.train()
                start = time.time()
                train_loss, n_samples = 0, 0
                for batch_idx, data in enumerate(train_loader):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                    optimizer.zero_grad()
                    if model_name == 'ExpandedSpatialGraphEmbeddingNet':
                        output = model(data, fold_id, 'train')
                    else:
                        output = model(data)
                    loss = loss_fn(output, data[4])
                    loss.backward()
                    optimizer.step()
                    time_iter = time.time() - start
                    total_time_iter += time_iter
                    train_loss += loss.item() * len(output)
                    n_samples += len(output)
                    if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                            epoch, n_samples, len(train_loader.dataset),
                            100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))
                scheduler.step()
                return total_time_iter / (batch_idx + 1)
            
            # Test function
            def test(test_loader):
                print('Test model ...')
                model.eval()
                start = time.time()
                test_loss, correct, n_samples = 0, 0, 0
                for batch_idx, data in enumerate(test_loader):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                    if model_name == 'ExpandedSpatialGraphEmbeddingNet':
                        output = model(data, fold_id, 'test')
                    else:
                        output = model(data)
                    loss = loss_fn(output, data[4], reduction='sum')
                    test_loss += loss.item()
                    n_samples += len(output)
                    pred = output.detach().cpu().max(1, keepdim=True)[1]
    
                    correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
    
                time_iter = time.time() - start
    
                test_loss /= n_samples
    
                acc = 100. * correct / n_samples
                print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                                                                                                      test_loss, 
                                                                                                      correct, 
                                                                                                      n_samples, acc))
                return acc
            
            # Loss function
            loss_fn = F.cross_entropy
            
            total_time = 0
            for epoch in range(args.epochs):
                total_time_iter = train(loaders[0])
                total_time += total_time_iter
                acc = test(loaders[1])
            acc_folds.append(round(acc,2))
            time_folds.append(round(total_time/args.epochs,2))
            
            # Save model
            if args.save_model:
                print('Save model ...')
                create_directory('./save_model')
                create_directory('./save_model/' + model_name)
                
                # Guardam els models en un arxiu .pt que serveix per lletgir-lo amb el paquet torch.
                file_name = model_name + '_' + dataset_name + '_' + readout_name + '_' + str(fold_id) + '_' + str(args.n_agg_layer) + '_h' + str(args.agg_hidden) + '.pt'

                torch.save(model, './save_model/' + model_name + '/' + file_name)
                '''
                Els arxius .pt els podem lletgir de manera sencilla amb torch:
                    > import torch
                    > # Definim un model qualsevol igual que el que volem carregar
                    > model = MyModel()
                    > model.load_state_dict(torch.load("model.pt"))
                '''

                print('Complete to save model')

        print(acc_folds)
        print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds, statistics.mean(acc_folds), statistics.stdev(acc_folds)))

        # Save 10-cross validation result as csv format
        create_directory('./test_result')
        create_directory('./test_result/' + model_name)
        
        result_list = []
        result_list.append(dataset_name)
        result_list.append(readout_name)
        for acc_fold in acc_folds:
          result_list.append(str(acc_fold))
        result_list.append(statistics.mean(acc_folds))
        result_list.append(statistics.stdev(acc_folds))
        result_list.append(statistics.mean(time_folds))
                         

        file_name = model_name + '_' + str(args.n_agg_layer) + '_h' + str(args.agg_hidden) + '_' + '10_cross_validation.csv'
        if args.n_graph_subsampling > 0 and args.graph_node_subsampling:
          file_name = model_name + '_' + str(args.n_agg_layer) + '_h' + str(args.agg_hidden) + '_' + 'node_graph_subsampling' + '_' + '10_cross_validation.csv'
        elif args.n_graph_subsampling > 0:
          file_name = model_name + '_' + str(args.n_agg_layer) + '_h' + str(args.agg_hidden) + '_' + 'edge_graph_subsampling' + '_' + '10_cross_validation.csv'
        
        if i == 0:
          save_result_csv('./test_result/' + model_name + '/' + file_name, result_list, True)
        else:
          save_result_csv('./test_result/' + model_name + '/' + file_name, result_list, False)
        
        print('-'*25)
    print('-'*50)
