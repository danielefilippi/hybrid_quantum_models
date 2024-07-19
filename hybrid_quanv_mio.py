import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch

from hqm.circuits.angleencoding import BasicEntangledCircuit
from hqm.layers.basiclayer import BasicLayer
from hqm.classification.hcnn import HybridLeNet5
import pennylane as qml
import sys
class HybridNet(pl.LightningModule):

    def __init__(self):
        super(HybridNet, self).__init__()
        dev1 = qml.device("lightning.qubit", wires=16)
        dev2 = qml.device("lightning.qubit", wires=4)
        
        qcircuit_1 = BasicEntangledCircuit(n_qubits=16, n_layers=1, dev=dev1)
        qlayer_1 = Quanvolution2D(qcircuit_1, filters=3, kernelsize=2, stride=1, padding='same', aiframework='torch')
        qcircuit_2 = BasicEntangledCircuit(n_qubits=4, n_layers=2, dev=dev2)
        qlayer_2 = BasicLayer(qcircuit_2, aiframework='torch')
        
        
        self.network = HybridLeNet5_quanv_2(qlayer_1= qlayer_1, qlayer_2=qlayer_2, in_shape=(3, 64, 64), ou_dim=10)
        self.loss = torch.nn.CrossEntropyLoss()
        

    def forward(self, x):
        return self.network.forward(x)

        
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        #print(labels, outputs)

        loss      = self.loss(outputs, labels)
        # For example, log accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = torch.sum(predicted == labels.data).item() / labels.size(0)

        # Logging info
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        
        loss      = self.loss(outputs, labels)
        # For example, log accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = torch.sum(predicted == labels.data).item() / labels.size(0)

        # Logging info
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
      from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from dataio_tot_classes.loader import EuroSATDataModule
from metrics import MetricsLogger

# Definisco i paths delle mie directory
base_path = "C:/Users/danfi/anaconda3/envs/qml/QML-tutorial"
dataset_path = os.path.join(base_path, "dataset")
train_dir = os.path.join(dataset_path, "training")
val_dir = os.path.join(dataset_path, "validation")

# Ora definisco EuroSATDataModule con i path del mio dataset, EuroSATDataModule è una sottoclasse pl.LightningDataModule
data_module = EuroSATDataModule(train_dir=train_dir, val_dir=val_dir,batch_size=8, num_workers=16)

# definisco TensorBoard logger per registrare i log di addestramento e convalida 
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError("Assicurati che le directory del dataset esistano e siano corrette")

# Definisci EuroSATDataModule con i path del dataset
data_module = EuroSATDataModule(train_dir=train_dir, val_dir=val_dir, batch_size=8, num_workers=16)

# Definisci TensorBoard logger per registrare i log di addestramento e convalida
log_dir = os.path.join(base_path, 'lightning_logs_hyb_tot', 'classifiers', 'EuroSATClassifier')
tb_logger = pl.loggers.TensorBoardLogger(log_dir)
# Instantiate ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join('saved_models__hyb_tot','classifiers'),
    filename='EuroSATClassifier',
    monitor='val_loss',
    save_top_k=1,
    mode='min',
)
#richiamo la funzione che ho fatto io
metrics_logger = MetricsLogger()
# Selezione del dispositivo perche voglio usare cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)
# Instantiate LightningModule and DataModule
model = HybridNet()

model.to(device)


trainer = pl.Trainer(max_epochs=5,callbacks=[checkpoint_callback, metrics_logger], logger=tb_logger, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)



trainer.fit(model, data_module)
# quindi lancia: tensorboard --logdir=C:/Users/danfi/anaconda3/envs/qml/QML-tutorial/lightning_logs/classifiers/EuroSATClassifier
# Dopo l'allenamento, puoi accedere ai valori memorizzati come segue:
train_losses = metrics_logger.train_losses
val_losses = metrics_logger.val_losses
train_accuracies = metrics_logger.train_accuracies
val_accuracies = metrics_logger.val_accuracies

# Stampa i valori per verificarne il contenuto
print("Train Losses:", train_losses)
print("Validation Losses:", val_losses)
print("Train Accuracies:", train_accuracies)
print("Validation Accuracies:", val_accuracies)
import matplotlib.pyplot as plt

# Supponiamo che i seguenti dati siano stati ottenuti dall'addestramento del tuo modello
epochs = list(range(1, len(train_losses) + 1))

# Plot per Loss (Training e Validation) vs Epoch
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='red')
plt.plot(epochs, val_losses[:len(epochs)], label='Validation Loss', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plot per Accuracy (Training e Validation) vs Epoch
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o', color='red')
plt.plot(epochs, val_accuracies[:len(epochs)], label='Validation Accuracy', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
#calcolo max e min accuraratezza/loss
max_train_accuracy = max(train_accuracies)
min_train_loss = min(train_losses)
max_val_accuracy = max(val_accuracies[:len(epochs)])  # Considera solo i valori corrispondenti alle epoche effettive
min_val_loss = min(val_losses[:len(epochs)])  # Considera solo i valori corrispondenti alle epoche effettive

print("Massimo dell'accuratezza di training:", max_train_accuracy)
print("Minimo della loss di training:", min_train_loss)
print("Massimo dell'accuratezza di validazione:", max_val_accuracy)
print("Minimo della loss di validazione:", min_val_loss)
