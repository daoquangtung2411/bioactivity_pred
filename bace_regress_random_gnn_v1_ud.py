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
from helper.load_dataset import load_bace_regression
from tabulate import tabulate
from helper.preprocess import split_train_valid_test, generate_graph_dataset
from helper.trainer import fit_model, evaluate_test, final_fit_model, final_evaluate
from helper.graphfeat import StructureEncoderV1, StructureEncoderV2, StructureEncoderV3, StructureEncoderV4, StructureEncoderV5
from helper.cal_metrics import regression_metrics

bace = load_bace_regression()
train, valid, test = split_train_valid_test(bace)


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

    metrics = evaluate_test(model, valid_loader)
    rmse = metrics['rmse']
    return {
        'loss': rmse,
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
        # task='classification',
        use_edge_weight=best_params['use_edge_weight']
    )
    
    train_stats = final_evaluate(best_model, merge_loader)
    test_stats = final_evaluate(best_model, test_loader)

    return train_stats, test_stats

print('V1')

encoder = StructureEncoderV1(directed=False) # directed = False mean undirected graph

train_stats, test_stats = run_gcn_tuning(train, valid, test, encoder, max_evals=100)

train_metrics = regression_metrics(train_stats['y_true'], train_stats['y_pred'])
test_metrics = regression_metrics(test_stats['y_true'], test_stats['y_pred'])

result_header = ['Metrics', 'Train', 'Test']

result_body = [
    ["RMSE", f"{train_metrics['rmse']:.4f}", f"{test_metrics['rmse']:.4f}"],
    ["R2", f"{train_metrics['r2']:.4f}", f"{test_metrics['r2']:.4f}"],
    ["MAPE", f"{train_metrics['mape']:.4f}", f"{test_metrics['mape']:.4f}"],
    ["Pearson R", f"{train_metrics['pearsonr']:.4f}", f"{test_metrics['pearsonr']:.4f}"],
    ["Spearman R", f"{train_metrics['spearmanr']:.4f}", f"{test_metrics['spearmanr']:.4f}"],
]

print(tabulate(result_body, headers=result_header, tablefmt='grid'))

with open('results/bace_regress_gnn_random_v1.txt', 'w') as file:
    file.write(f'BACE classfication\n')
    file.write('ANN Classifier results:\n')
    # file.write(f'Best params: {best_ann_params}')
    file.write(tabulate(result_body, headers=result_header, tablefmt='grid'))

print('V2')

encoder = StructureEncoderV2(directed=False) # directed = False mean undirected graph

train_stats, test_stats = run_gcn_tuning(train, valid, test, encoder, max_evals=100)

train_metrics = regression_metrics(train_stats['y_true'], train_stats['y_pred'])
test_metrics = regression_metrics(test_stats['y_true'], test_stats['y_pred'])

result_header = ['Metrics', 'Train', 'Test']

result_body = [
    ["RMSE", f"{train_metrics['rmse']:.4f}", f"{test_metrics['rmse']:.4f}"],
    ["R2", f"{train_metrics['r2']:.4f}", f"{test_metrics['r2']:.4f}"],
    ["MAPE", f"{train_metrics['mape']:.4f}", f"{test_metrics['mape']:.4f}"],
    ["Pearson R", f"{train_metrics['pearsonr']:.4f}", f"{test_metrics['pearsonr']:.4f}"],
    ["Spearman R", f"{train_metrics['spearmanr']:.4f}", f"{test_metrics['spearmanr']:.4f}"],
]

print(tabulate(result_body, headers=result_header, tablefmt='grid'))

with open('results/bace_regress_gnn_random_v2.txt', 'w') as file:
    file.write(f'BACE classfication\n')
    file.write('ANN Classifier results:\n')
    # file.write(f'Best params: {best_ann_params}')
    file.write(tabulate(result_body, headers=result_header, tablefmt='grid'))

print('V3')

encoder = StructureEncoderV3(directed=False) # directed = False mean undirected graph

train_stats, test_stats = run_gcn_tuning(train, valid, test, encoder, max_evals=100)

train_metrics = regression_metrics(train_stats['y_true'], train_stats['y_pred'])
test_metrics = regression_metrics(test_stats['y_true'], test_stats['y_pred'])

result_header = ['Metrics', 'Train', 'Test']

result_body = [
    ["RMSE", f"{train_metrics['rmse']:.4f}", f"{test_metrics['rmse']:.4f}"],
    ["R2", f"{train_metrics['r2']:.4f}", f"{test_metrics['r2']:.4f}"],
    ["MAPE", f"{train_metrics['mape']:.4f}", f"{test_metrics['mape']:.4f}"],
    ["Pearson R", f"{train_metrics['pearsonr']:.4f}", f"{test_metrics['pearsonr']:.4f}"],
    ["Spearman R", f"{train_metrics['spearmanr']:.4f}", f"{test_metrics['spearmanr']:.4f}"],
]

print(tabulate(result_body, headers=result_header, tablefmt='grid'))

with open('results/bace_regress_gnn_random_v3.txt', 'w') as file:
    file.write(f'BACE classfication\n')
    file.write('ANN Classifier results:\n')
    # file.write(f'Best params: {best_ann_params}')
    file.write(tabulate(result_body, headers=result_header, tablefmt='grid'))

print('V4')

encoder = StructureEncoderV4(directed=False) # directed = False mean undirected graph

train_stats, test_stats = run_gcn_tuning(train, valid, test, encoder, max_evals=100)

train_metrics = regression_metrics(train_stats['y_true'], train_stats['y_pred'])
test_metrics = regression_metrics(test_stats['y_true'], test_stats['y_pred'])

result_header = ['Metrics', 'Train', 'Test']

result_body = [
    ["RMSE", f"{train_metrics['rmse']:.4f}", f"{test_metrics['rmse']:.4f}"],
    ["R2", f"{train_metrics['r2']:.4f}", f"{test_metrics['r2']:.4f}"],
    ["MAPE", f"{train_metrics['mape']:.4f}", f"{test_metrics['mape']:.4f}"],
    ["Pearson R", f"{train_metrics['pearsonr']:.4f}", f"{test_metrics['pearsonr']:.4f}"],
    ["Spearman R", f"{train_metrics['spearmanr']:.4f}", f"{test_metrics['spearmanr']:.4f}"],
]

print(tabulate(result_body, headers=result_header, tablefmt='grid'))

with open('results/bace_regress_gnn_random_v4.txt', 'w') as file:
    file.write(f'BACE classfication\n')
    file.write('ANN Classifier results:\n')
    # file.write(f'Best params: {best_ann_params}')
    file.write(tabulate(result_body, headers=result_header, tablefmt='grid'))

print('V5')

encoder = StructureEncoderV5(directed=False) # directed = False mean undirected graph

train_stats, test_stats = run_gcn_tuning(train, valid, test, encoder, max_evals=100)

train_metrics = regression_metrics(train_stats['y_true'], train_stats['y_pred'])
test_metrics = regression_metrics(test_stats['y_true'], test_stats['y_pred'])

result_header = ['Metrics', 'Train', 'Test']

result_body = [
    ["RMSE", f"{train_metrics['rmse']:.4f}", f"{test_metrics['rmse']:.4f}"],
    ["R2", f"{train_metrics['r2']:.4f}", f"{test_metrics['r2']:.4f}"],
    ["MAPE", f"{train_metrics['mape']:.4f}", f"{test_metrics['mape']:.4f}"],
    ["Pearson R", f"{train_metrics['pearsonr']:.4f}", f"{test_metrics['pearsonr']:.4f}"],
    ["Spearman R", f"{train_metrics['spearmanr']:.4f}", f"{test_metrics['spearmanr']:.4f}"],
]

print(tabulate(result_body, headers=result_header, tablefmt='grid'))

with open('results/bace_regress_gnn_random_v5.txt', 'w') as file:
    file.write(f'BACE classfication\n')
    file.write('ANN Classifier results:\n')
    # file.write(f'Best params: {best_ann_params}')
    file.write(tabulate(result_body, headers=result_header, tablefmt='grid'))
