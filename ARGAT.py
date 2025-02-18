import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from LoadTrainData import LoadTrainData
from LoadTestData import LoadTestData
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Setting the Path
train_path = "xxx" # Path to the training dataset
test_path = "xxx"
train_dataset = "xxx..." # filename
test_dataset = "xxx..."
np.random.seed(400)

n_features = 512
epochs = 200
early_stop_patience = 5
weight_decay = 1e-4
learning_rate = 0.00001
batch_size = 64

# Loading training and testing data
tx, ty, ta = LoadTestData(test_path, test_dataset)
print('tx', tx.shape)
X_a, A_a = LoadTrainData(train_path, train_dataset)
num_nodes = tx.shape[1]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tx = tx.to(device)
ty = ty.to(device)
ta = ta.to(device)


# Defining the model class
class GConvolution(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, bias: float = 0.0):
        super(GConvolution, self).__init__()
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features * self.num_nodes, self.num_nodes * out_features)
        )
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.num_nodes * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, adj):
        bat_long = inputs.size(0)
        inputs = torch.reshape(inputs, (-1, self.num_nodes, self.in_features))
        support = torch.bmm(adj, inputs)
        support = torch.reshape(support, (bat_long, self.num_nodes * self.in_features))
        output = torch.mm(support, self.weight)
        outputs = torch.reshape(output, (bat_long, self.out_features, self.num_nodes))
        if self.use_bias:
            return outputs + self.bias
        else:
            return outputs



class TConvolution(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, bias: float = 0.0):
        super(TConvolution, self).__init__()
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features * self.num_nodes, self.in_features * out_features)
        )
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.num_nodes * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, adj):
        fg = torch.bmm(inputs, adj)
        temp = torch.reshape(fg, (-1, self.num_nodes * self.in_features))
        te = torch.mm(temp, self.weight)
        outputs = torch.reshape(te, (-1, self.in_features, self.out_features))
        if self.use_bias:
            return outputs + self.bias
        else:
            return outputs

#Building the ARGAT model
class ARGAT(nn.Module):
    def __init__(self, seq_len, num_nodes, f_out, gat_out_size, num_heads, out_size):
        super(ARGAT, self).__init__()
        self.out_size = out_size
        self.f_out = f_out
        self.seq_long = seq_len
        self._num_nodes = num_nodes
        self.gat_out_size = gat_out_size
        self.num_heads = num_heads
        self.gcn1 = GConvolution(seq_len, num_nodes, f_out)
        self.gat1 = GATConv(f_out, gat_out_size, heads=num_heads, concat=True, dropout=0.5)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.50)
        self.Tcn1 = TConvolution(gat_out_size * num_heads, num_nodes, out_size)


        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, 512)  # Note the dimension here
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, X, adj):
        G1 = F.gelu(self.gcn1(X, adj))

        G1 = self.dropout(G1)

        # The input of the GAT layer should be (batch_size, num_nodes, f_out)
        G1 = G1.permute(0, 2, 1)  # Transform the dimensions to (batch_size, num_nodes, f_out)

        # Convert to torch_geometric Data object
        data_list = []
        for i in range(G1.size(0)):
            edge_index = adj[i].nonzero(as_tuple=False).t().contiguous()  # Get edge_index of [2, num_edges]
            data = Data(x=G1[i], edge_index=edge_index)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)

        G2 = F.elu(self.gat1(batch.x, batch.edge_index))
        G2 = G2.view(-1, self._num_nodes, self.gat_out_size * self.num_heads)  # Adjust dimensions
        G2 = self.dropout(G2)

        # Restore the dimensions to (batch_size, gat_out_size * num_heads, num_nodes)
        G2 = G2.permute(0, 2, 1)  # Swap dimensions to accommodate subsequent operations

        T1 = F.tanh(self.Tcn1(G2, adj))
        T1 = self.dropout(T1)

        s = torch.reshape(T1, (T1.size(0), -1))

        out = self.fc(s)
        out = F.selu(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.selu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)



model = ARGAT(512, 40, 256,32, num_heads=4, out_size=16)
model = model.to(device)
criterion = nn.CrossEntropyLoss(reduction='none').to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
torch.cuda.empty_cache()
best_val_acc = 0
patience_counter = 0

# Use generator_data to generate the entire training dataset
train_x, train_y, train_adjs = [], [], []
for x, y, adjs in DataGenerator(train_path, train_dataset, batch_size, X_a, A_a):
    train_x.append(x)
    train_y.append(y)
    train_adjs.append(adjs)
train_x = torch.cat(train_x)
train_y = torch.cat(train_y)
train_adjs = torch.cat(train_adjs)

train_ids = torch.randperm(train_x.size(0))
train_loader = DataLoader(
    TensorDataset(train_x[train_ids], train_y[train_ids], train_adjs[train_ids]),
    batch_size=batch_size, shuffle=True, drop_last=False
)

# train the model
train_loss_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0

    for x_batch, y_batch, adj_batch in train_loader:
        x_batch, y_batch, adj_batch = x_batch.to(device), y_batch.to(device), adj_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch, adj_batch)
        loss = criterion(output, y_batch).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        _, predicted = torch.max(output, 1)
        correct += (predicted == y_batch).sum().item()

    train_loss = total_loss / len(train_loader.dataset)
    train_loss_history.append(train_loss)
    train_acc = correct / len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        output = model(tx, ta)
        val_loss = criterion(output, ty).mean().item()
        val_loss_history.append(val_loss)
        _, predicted = torch.max(output, 1)
        correct = (predicted == ty).sum().item()
        val_acc = correct / ty.size(0)
        val_acc_history.append(val_acc)

    print(
        f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Using a scheduler to adjust the learning rate
    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print("Early stopping...")
        break

# Load the best model and evaluate it on the test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    test_output = model(tx, ta)
    _, test_predicted = torch.max(test_output, 1)
    test_correct = (test_predicted == ty).sum().item()
    test_acc = test_correct / ty.size(0)
    test_loss = criterion(test_output, ty).mean().item()

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# Plot the loss and accuracy history
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss History')

plt.subplot(1, 2, 2)
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy History')

plt.tight_layout()
plt.show()


labels = ['PD', 'BPSK', 'CW', 'LFMPC', 'SFMW', 'SINFM', 'TRIFM']

# Create a confusion matrix
cm = confusion_matrix(ty.cpu().numpy(), test_predicted.cpu().numpy())


sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
