import numpy as np
import matplotlib.pyplot as plt
# import torch
from portiloop_ml.portiloop_python.ANN.lightning_tests import SleepStagingModel
from portiloop_ml.portiloop_python.ANN.data.sleepedf_data import get_sleepedf_loaders_keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout, LSTM, Dense, Softmax, Flatten

config = {
    'batch_size': 64,
    'freq': 100,
    'inception': [16, 8, 16, 16, 32, 16],
    'lr': 1e-3,
    'num_heads': 8,
    'num_layers': 1,
    'noise_std': 0.1,
    'dropout': 0.1,
    'cls': False,
    'window_size': 30 * 100,
    'seq_len': 1,
}

# model_test = SleepStagingModel(config, [])
# model_test.load_state_dict(torch.load(
#     '/home/ubuntu/portiloop-training/TSN_again_1688500831-epoch=02-f1=0.71.ckpt')['state_dict'])


def TinySleepNet(fs):
    input_shape = (30 * fs, 1)
    inputs = Input(shape=input_shape)

    x = Conv1D(128, 16, strides=4)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=8, strides=8)(x)
    x = Dropout(0.5)(x)

    x = Conv1D(128, 8, 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(128, 8, 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(128, 8, 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(0.5)(x)

    # x = LSTM(64, name='embedding')(x)
    # x = ReLU()(x)

    x = Flatten()(x)

    outputs = Dense(5)(x)
    outputs = Softmax()(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


print("Loading data...")
train_loader, test_loader = get_sleepedf_loaders_keras(82, config)

print("Initializing model...")
model = TinySleepNet(100)
optim = tf.keras.optimizers.AdamW(learning_rate=1e-3)
model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=[
              'accuracy'])  # , 'f1_score', 'precision', 'recall'


# print(next(train_loader)[0].shape)

print("Training model...")
model.fit(train_loader, epochs=100, validation_data=test_loader,
          batch_size=64, verbose=1, steps_per_epoch=1000, validation_steps=1000)

test = next(train_loader)
preds = model.predict(test[0])
print(preds)
