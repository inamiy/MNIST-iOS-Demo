from torch.autograd import Variable
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import onnx
import onnx.helper
from onnx_coreml import convert

num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5),    # 28x28x1 -> 24x24x10
    nn.MaxPool2d(kernel_size=2),        # -> 12x12x10
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=5),   # -> 8x8x20
    nn.Dropout2d(),
    nn.MaxPool2d(kernel_size=2),        # -> 4x4x20
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(4 * 4 * 20, 50),
    nn.Dropout2d(),
    nn.Linear(50, 10),

    # Comment-Out: CrossEntropy will run instead.
    # nn.LogSoftmax()
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
if True:
    num_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"epoch {epoch + 1} / {num_epochs}, step {i + 1} / {num_total_steps}, loss = {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")
else:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

# Test
if True:
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # max returns `value, index`
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]  # 100
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f"accuracy = {acc}")

# Save as ONNX
dummy = Variable(torch.randn(1, 1, 28, 28))
torch.onnx.export(model, dummy, 'model.onnx', verbose=True)

# Load ONNX
model2 = onnx.load("model.onnx")
# print(onnx.helper.printable_graph(model2.graph))

# Make CoreML from ONNX
# https://medium.com/@kuluum/pytroch-to-coreml-cheatsheet-fda57979b3c6
model_coreml = convert(model='model.onnx')

# Save CoreML model
model_coreml.save('Model.mlmodel')
