import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_max_pool, global_mean_pool, global_add_pool
from hyperopt import hp, tpe, fmin, Trials, space_eval
from hyperopt.pyll import scope
from helper.load_dataset import load_bace_classification
from tabulate import tabulate
from helper.preprocess import split_train_valid_test, generate_graph_dataset
from helper.trainer import fit_model, evaluate_test, final_fit_model, final_evaluate
from helper.graphfeat import StructureEncoderV1
from helper.cal_metrics import classification_metrics

bace = load_bace_classification()
train, valid, test = split_train_valid_test(bace)

encoder = StructureEncoderV1(directed=False) # directed = False mean undirected graph
class GraphConvClassifier(nn.Module):
    def __init__(
            self,
            num_node_features,
            hidden_channels=64,
            num_layers=3,
            dropout_rate=0.2,
            pooling='max',
            use_edge_weight=True,
            num_linear_layers = 2,
            linear_hidden_1=32,
            linear_hidden_2=16,
            activation='relu'
    ):
        super(GraphConvClassifier, self).__init__()
        self.use_edge_weight = use_edge_weight
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.pooling = pooling
        self.num_linear_layers=num_linear_layers
        self.activation = activation

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(num_node_features, hidden_channels, bias=True))
        for _ in range(num_layers-1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, bias=True))
        
        self.linears = nn.ModuleList()
        if num_linear_layers == 1:
            self.linears.append(nn.Linear(hidden_channels, 1))
        elif num_linear_layers == 2:
            self.linears.append(nn.Linear(hidden_channels, linear_hidden_1))
            self.linears.append(nn.Linear(linear_hidden_1, 1))
        else:
            self.linears.append(nn.Linear(hidden_channels, linear_hidden_1))
            self.linears.append(nn.Linear(linear_hidden_1, linear_hidden_2))
            self.linears.append(nn.Linear(linear_hidden_2, 1))
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch, edge_weight=None):
        use_ew = edge_weight if self.use_edge_weight else None

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=use_ew)
            x = self._activation(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        
        if self.pooling == 'max':
            x = global_max_pool(x, batch)

        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_add_pool(x, batch)

        for i, lin in enumerate(self.linears[:-1]):
            x = lin(x)
            x = self._activation(x)
            x = self.dropout(x)
        
        x = self.linears[-1](x)

        return x

    def _activation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'gelu':
            return F.gelu(x)
        elif self.activation == 'elu':
            return F.elu(x)
        elif self.activation == 'selu':
            return F.selu(x)
        else:
            return F.relu(x)
    
def gcn_objective(
        params,
        train_dataset,
        valid_dataset,
        num_node_features
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=params['batch_size'],
        shuffle=False
    )

    model = GraphConvClassifier(
        num_node_features=num_node_features,
        hidden_channels=params['hidden_channels'],
        num_layers=params['num_layers'],
        dropout_rate=params['dropout_rate'],
        pooling=params['pooling'],
        use_edge_weight=params['use_edge_weight'],
        num_linear_layers=params['num_linear_layers'],
        linear_hidden_1=params['linear_hidden_1'],
        linear_hidden_2=params['linear_hidden_2'],
        activation=params['activation']
    )

    history = fit_model(
        model,
        train_loader,
        valid_loader,
        epochs=params['epochs'],
        lr=params['learning_rate'],
        patience=params['patience'],
        task='classification',
        use_edge_weight=params['use_edge_weight']
    )

    metrics = evaluate_test(model, valid_loader, task='classification')
    f1 = metrics['f1']
    return {
        'loss': -f1,
        'status': 'ok',
        'best_num_epoch': len(history)
    }

gcn_search_space = {
    'hidden_channels': scope.int(hp.quniform('hidden_channels', 32, 128, 16)),
    'num_layers': scope.int(hp.quniform('num_layers', 2, 5, 1)),
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'batch_size': scope.int(hp.quniform('batch_size', 32, 256, 32)),
    'num_linear_layers': scope.int(hp.quniform('num_linear_layers', 1, 3, 1)),
    'linear_hidden_1': scope.int(hp.quniform('linear_hidden_1', 16, 64, 8)),
    'linear_hidden_2': scope.int(hp.quniform('linear_hidden_2', 8, 32, 4)),
    'pooling': hp.choice('pooling', ['max', 'mean']),
    'use_edge_weight': hp.choice('use_edge_weight', [True, False]),
    'activation': hp.choice('activation', ['relu', 'selu', 'elu', 'gelu']),
    'epochs': 200,
    'patience': 15
}


def run_gcn_tuning(train_data, valid_data, test_data, encoder, max_evals=100):
    
    # Generate graph datasets
    
    train_dataset = generate_graph_dataset(train_data, 'SMILES', 'Class', encoder=encoder)
    valid_dataset = generate_graph_dataset(valid_data, 'SMILES', 'Class', encoder=encoder)
    test_dataset = generate_graph_dataset(test_data, 'SMILES', 'Class', encoder=encoder)
    
    # Run hyperparameter optimization
    num_node_features = train_dataset.num_node_features
    objective_fn = lambda params: gcn_objective(
        params, 
        train_dataset,
        valid_dataset,
        num_node_features
    )
    
    trials = Trials()
    best_params = fmin(
        fn=objective_fn,
        space=gcn_search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    best_params = space_eval(gcn_search_space, best_params)
    print(f"\nBest parameters: {best_params}")
    
    # Train final model with best parameters
    best_model = GraphConvClassifier(
        num_node_features=num_node_features,
        hidden_channels=best_params['hidden_channels'],
        num_layers=best_params['num_layers'],
        dropout_rate=best_params['dropout_rate'],
        pooling=best_params['pooling'],
        use_edge_weight=best_params['use_edge_weight'],
        num_linear_layers=best_params['num_linear_layers'],
        linear_hidden_1=best_params['linear_hidden_1'],
        linear_hidden_2=best_params['linear_hidden_2'],
        activation=best_params['activation']
    )
    
    merge_data = pd.concat([train_data, valid_data], ignore_index=True)
    merge_dataset = generate_graph_dataset(merge_data, 'SMILES', 'Class', encoder=encoder)

    # Create data loaders with best batch size
    merge_loader = DataLoader(
        merge_dataset, 
        batch_size=best_params['batch_size'], 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=best_params['batch_size'], 
        shuffle=False
    )
    
    # Get best epoch from trials
    best_trial = trials.best_trial
    best_num_epochs = best_trial['result']['best_num_epoch']
    
    print(f"\nTraining final model for {best_num_epochs} epochs...")
    history = final_fit_model(
        best_model,
        merge_loader,
        epochs=best_num_epochs,
        lr=best_params['learning_rate'],
        task='classification',
        use_edge_weight=best_params['use_edge_weight']
    )
    
    train_stats = final_evaluate(best_model, merge_loader, task='classification')
    test_stats = final_evaluate(best_model, test_loader, task='classification')

    return train_stats, test_stats

train_stats, test_stats = run_gcn_tuning(train, valid, test, encoder, max_evals=100)

train_metrics = classification_metrics(train_stats['y_true'], train_stats['y_pred'], train_stats['y_scores'])
test_metrics = classification_metrics(test_stats['y_true'], test_stats['y_pred'], test_stats['y_scores'])

result_header = ['Metrics', 'Train', 'Test']
result_body = [
    ["Accuracy", f"{train_metrics['accuracy']:.4f}", f"{test_metrics['accuracy']:.4f}"],
    ["Recall"],
    ["Overall recall", f"{train_metrics['recall']:.4f}", f"{test_metrics['recall']:.4f}"],
    ["Class 0 recall", f"{train_metrics['0_recall']:.4f}", f"{test_metrics['0_recall']:.4f}"],
    ["Class 1 recall", f"{train_metrics['1_recall']:.4f}", f"{test_metrics['1_recall']:.4f}"],
    ["Precision", "", ""],
    ["Overall precision", f"{train_metrics['precision']:.4f}", f"{test_metrics['precision']:.4f}"],
    ["Class 0 precision", f"{train_metrics['0_precision']:.4f}", f"{test_metrics['0_precision']:.4f}"],
    ["Class 1 precision", f"{train_metrics['1_precision']:.4f}", f"{test_metrics['1_precision']:.4f}"],
    ["AUC-ROC", f"{train_metrics['auc-roc']:.4f}", f"{test_metrics['auc-roc']:.4f}"],
    ["AUC-PRC", f"{train_metrics['auc-prc']:.4f}", f"{test_metrics['auc-prc']:.4f}"],
]

# print('ANN Classifier results:')
# print(f'Best params: {best_ann_params}')
print(tabulate(result_body, headers=result_header, tablefmt='grid'))

with open('results/bace_class_gnn.txt', 'w') as file:
    file.write(f'BACE classfication\n')
    file.write('ANN Classifier results:\n')
    # file.write(f'Best params: {best_ann_params}')
    file.write(tabulate(result_body, headers=result_header, tablefmt='grid'))
