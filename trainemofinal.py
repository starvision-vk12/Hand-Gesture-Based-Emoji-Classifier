import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

# Define the model
class CNN(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=5, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    data = pd.read_csv("train_foo.csv")
    # b = len(data)
    # print(b)
    dataset = np.array(data)
    # a = len(dataset)
    # print(a)
    np.random.shuffle(dataset)
    X = dataset[:, 1:2501] / 255.0
    Y = dataset[:, 0]

    # Convert to tensors
    X = torch.tensor(X.reshape(-1, 1, 50, 50), dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)

    # Split the data
    dataset_size = len(X)
    print("DATA:-",dataset_size)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.20 * dataset_size)
    test_size = dataset_size - train_size - val_size

    print("After spliting train-data size",train_size)
    print("After spliting validation-data size",val_size)
    print("After spliting testing-data size",test_size)

    train_dataset, val_dataset, test_dataset = random_split(TensorDataset(X, Y), [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNN(num_classes=12)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
        
        # Evaluating on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += Y_batch.size(0)
                correct += (predicted == Y_batch).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Evaluate the model on the test set
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(Y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(outputs.cpu().numpy())
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

    print(f"Validation Accuracy of the model on the test images: {100 * correct / total:.2f}%")

    #saving the model
    torch.save(model.state_dict(), 'emojinator.pth')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Classification Report
    cr = classification_report(y_true, y_pred)
    print("Classification Report:\n", cr)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate ROC AUC
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(12):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting ROC curve for each class
    plt.figure(figsize=(14, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy', 'darkgreen', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    for i, color in zip(range(12), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
