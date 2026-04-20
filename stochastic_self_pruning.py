
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hard_concrete_sample(log_alpha, beta=0.2):
    u = torch.rand_like(log_alpha)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    s_bar = s * 1.2 - 0.1
    return torch.clamp(s_bar, 0, 1)

class StochasticPrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.log_alpha = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = hard_concrete_sample(self.log_alpha)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

class StochasticPrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = StochasticPrunableLinear(32 * 32 * 3, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = StochasticPrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = StochasticPrunableLinear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if isinstance(module, StochasticPrunableLinear):
            gates = torch.sigmoid(module.log_alpha)
            loss += gates.sum()
    return loss

def train_model(lambda_val):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128)

    model = StochasticPrunableNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            cls_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)
            loss = cls_loss + lambda_val * sp_loss / images.size(0)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total

    total_weights, pruned = 0, 0
    all_gates = []
    for module in model.modules():
        if isinstance(module, StochasticPrunableLinear):
            gates = torch.sigmoid(module.log_alpha)
            total_weights += gates.numel()
            pruned += (gates < 0.1).sum().item()
            all_gates.append(gates.detach().cpu().flatten())

    sparsity = 100 * pruned / total_weights
    all_gates = torch.cat(all_gates)

    return accuracy, sparsity, all_gates

lambdas = [0.05, 0.1, 0.2]
results = []

for lam in lambdas:
    print(f"Training with lambda = {lam}")
    acc, sp, gates = train_model(lam)
    results.append((lam, acc, sp))
    print(f"Accuracy: {acc:.2f}%, Sparsity: {sp:.2f}%")

plt.hist(gates.numpy(), bins=50)
plt.title("Stochastic Gate Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.savefig("stochastic_gate_distribution.png")
plt.show()

print("Final Results:")
for r in results:
    print(f"{r[0]} | {r[1]:.2f}% | {r[2]:.2f}%")
