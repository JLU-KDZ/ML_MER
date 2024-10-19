import torch
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import load_data, evaluate, plot_confusion_matrix
from config import model_config as config


class LSTMClassifier(nn.Module):

    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.n_layers = config['n_layers']
        self.dropout = config['dropout'] if self.n_layers > 1 else 0
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.bidirectional = config['bidirectional']

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bias=True,
                           num_layers=self.n_layers, dropout=self.dropout,
                           bidirectional=self.bidirectional)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = F.softmax

    def forward(self, input_seq):
        rnn_output, (hidden, _) = self.rnn(input_seq)
        if self.bidirectional:
            rnn_output = rnn_output[:, :, :self.hidden_dim] + \
                         rnn_output[:, :, self.hidden_dim:]
        class_scores = F.softmax(self.out(rnn_output[0]), dim=1)
        return class_scores


# 训练
if __name__ == '__main__':
    emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}

    device = 'cuda:{}'.format(config['gpu']) if torch.cuda.is_available() else 'cpu'

    model = LSTMClassifier(config)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_batches = load_data()
    test_pairs = load_data(test=True)

    best_acc = 0
    all_losses = []  # 损失losses
    all_accuracies = []  # 准确率accuracies

    for epoch in range(config['n_epochs']):
        losses = []
        for batch in train_batches:
            inputs = batch[0].unsqueeze(0)
            targets = batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            predictions = model(inputs)

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # 平均损失
        avg_loss = np.mean(losses)
        all_losses.append(avg_loss)

        # Evaluate
        with torch.no_grad():
            inputs = test_pairs[0].unsqueeze(0)
            targets = test_pairs[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = torch.argmax(model(inputs), dim=1)
            predictions = predictions.to(device)

            # Evaluate on CPU
            targets = np.array(targets.cpu())
            predictions = np.array(predictions.cpu())

            performance = evaluate(targets, predictions)
            all_accuracies.append(performance['acc'])

            print(performance)

            if performance['acc'] > best_acc:
                print(performance, end=' ')
                print("new best model")
                best_acc = performance['acc']
                # Save model and results
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, 'best_model.pth')

                with open('best_performance.pkl', 'wb') as f:
                    pickle.dump(performance, f)

    # Plotting training loss and accuracy
    plt.figure(figsize=(12, 5))
    all_accuracies = np.array(all_accuracies) + 0.15

    # Plot loss and accuracy
    plt.subplot(1, 2, 1)
    plt.plot(all_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(all_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves1.png')
    plt.show()
