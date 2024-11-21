import optuna
from optuna import Trial
import plotly
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import math
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.layers import Input, Dense, Dropout, BatchNormalization, Reshape, Flatten, Bidirectional, LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam

import optuna
from concurrent.futures import ProcessPoolExecutor
import sys
import logging

n_steps_in = 14  
n_steps_out = 5
epochs = 50
batch_size = 32

def load_processed_data(data_dir='processed_data'):
    """
    Load processed DataFrames list and numpy arrays from files
    """
    data_path = Path(data_dir)
    
    # Load list of DataFrames
    with open(data_path / 'processed_dfs.pkl', 'rb') as f:
        processed_dfs = joblib.load(f)
    
    # Load numpy arrays
    values_path = data_path / 'values_array.npy'
    values_list = np.load(values_path, allow_pickle=True)
    
    return processed_dfs, values_list

processed_dfs, values_list = load_processed_data('processed_data')

def preprocess_stock_data(values_list, n_steps_in=14, n_steps_out=5, train_split=0.8):
    # 1. Global scaling across all stocks
    global_scaler = MinMaxScaler(feature_range=(0, 1))
    combined_values = np.vstack(values_list)
    scaled_combined = global_scaler.fit_transform(combined_values)
    
    # 2. Split back into individual stocks
    scaled_values_list = []
    start_idx = 0
    for values in values_list:
        scaled_values_list.append(scaled_combined[start_idx:start_idx + len(values)])
        start_idx += len(values)
    
    # 3. Create sequences for each stock
    train_X_list, train_y_list = [], []
    val_X_list, val_y_list = [], []
    
    for scaled_values in scaled_values_list:
        # Remove 'Close Next Day' from features (last column)
        features = scaled_values[:, :-1]  # All columns except the last one
        targets = scaled_values[:, -1]    # Only the last column
        
        # Split into train/validation
        n_train = int(len(features) * train_split)
        
        # Ensure we have enough data for both training and validation
        if n_train <= n_steps_in + n_steps_out:
            print(f"Warning: Stock with {len(features)} samples is too short for meaningful splitting")
            continue
            
        # Split features and targets
        train_features = features[:n_train]
        train_targets = targets[:n_train]
        val_features = features[n_train:]
        val_targets = targets[n_train:]
        
        # Create sequences
        if len(train_features) > n_steps_in + n_steps_out:
            train_X, train_y = create_sequences(train_features, train_targets, n_steps_in, n_steps_out)
            train_X_list.append(train_X)
            train_y_list.append(train_y)
            
        if len(val_features) > n_steps_in + n_steps_out:
            val_X, val_y = create_sequences(val_features, val_targets, n_steps_in, n_steps_out)
            val_X_list.append(val_X)
            val_y_list.append(val_y)
    
    # 4. Combine all sequences
    train_X = np.vstack(train_X_list)
    train_y = np.vstack(train_y_list)
    val_X = np.vstack(val_X_list)
    val_y = np.vstack(val_y_list)
    
    print(f"Training shapes: X={train_X.shape}, y={train_y.shape}")
    print(f"Validation shapes: X={val_X.shape}, y={val_y.shape}")
    print(f"Number of features: {train_X.shape[2]}")
    
    return train_X, train_y, val_X, val_y, global_scaler

def create_sequences(features, targets, n_steps_in, n_steps_out):
    X, y = [], []
    
    # Ensure we have enough data for sequence creation
    if len(features) < n_steps_in + n_steps_out:
        raise ValueError("Data length is too short for the specified sequence lengths")
    
    for i in range(len(features) - n_steps_in - n_steps_out + 1):
        # Input sequence (n_steps_in days of all features)
        seq_x = features[i:(i + n_steps_in)]
        # Output sequence (next n_steps_out days of target variable)
        seq_y = targets[(i + n_steps_in):(i + n_steps_in + n_steps_out)]
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

train_X, train_y, val_X, val_y, scaler = preprocess_stock_data(values_list, n_steps_in, n_steps_out, train_split=0.8)

def configure_gpu():
    """Configure GPU memory growth and set memory limits"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Optionally, set memory limit (adjust the 4096 value based on your GPU memory)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                )
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")


# Enable Optuna performance optimizations
optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce logging overhead

def objective(trial):
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Hyperparameters to tune
    n_layers = trial.suggest_int('n_layers', 1, 3)  # Number of LSTM layers
    units = trial.suggest_int('units', 25, 200, log=True)  # Number of units per LSTM layer
    dropout = trial.suggest_float('dropout', 0.0, 0.5) 
    batch_size = trial.suggest_int('batch_size', 16, 64, log=True)

    # Create the LSTM model
    model = Sequential()
    for i in range(n_layers):
        return_sequences = i < n_layers - 1  # Return sequences for all layers except the last
        if i == 0:
            model.add(LSTM(units, activation='leaky_relu', dropout=dropout, 
                           return_sequences=return_sequences,
                           kernel_initializer='glorot_uniform',
                           input_shape=(train_X.shape[1], train_X.shape[2]),
                           name=f'lstm_layer_{i}'))
        else:
            model.add(LSTM(units, activation='leaky_relu', dropout=dropout, 
                           return_sequences=return_sequences,
                           kernel_initializer='glorot_uniform',
                           name=f'lstm_layer_{i}'))

    model.add(Dense(train_y.shape[1]))
    
    # Compile the model
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min', restore_best_weights=True)
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mae')
    
    
    # Train the model
    try:
        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
            use_multiprocessing=True,
            workers=4
        )
    
        # Return the minimum validation loss
        return min(history.history['val_loss'])
    
    except tf.errors.ResourceExhaustedError as e:
        print(f"GPU memory exhausted: {e}")
        trial.set_user_attr('error', 'gpu_memory_exhausted')
        raise optuna.exceptions.TrialPruned()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_optimization(process_id):
    """
    Run optimization with process identification
    """
    logger.info(f"Starting optimization process {process_id}")
    try:
        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///optuna.db",
            study_name='lstm_model_full',
            load_if_exists=True
        )
        
        study.optimize(
            objective, 
            n_trials=50, 
            gc_after_trial=True,
            show_progress_bar=True,
            catch=(Exception,)  # Catch exceptions to prevent silent failures
        )
        
        logger.info(f"Process {process_id} completed. Best value: {study.best_value}")
        return study.best_value
        
    except Exception as e:
        logger.error(f"Error in process {process_id}: {str(e)}")
        raise e

def run_parallel_optimization(n_processes=4):
    """
    Run multiple optimization processes in parallel
    """
    logger.info(f"Starting parallel optimization with {n_processes} processes")
    
    try:
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all processes and get futures
            futures = list(executor.map(run_optimization, range(n_processes)))
            
        # Process results
        for process_id, result in enumerate(futures):
            logger.info(f"Process {process_id} final best value: {result}")
            
    except Exception as e:
        logger.error(f"Error in parallel execution: {str(e)}")
        raise e

configure_gpu()
run_parallel_optimization(n_processes=4)