import torch
import matplotlib.pyplot as plt

from models import YOLOv1
from loss import SSELoss
from dataset import YOLOv1DataSet

from tqdm import tqdm
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

train_set = YOLOv1DataSet("data", image_set="train")
train_data = DataLoader(train_set, batch_size=10)

model = YOLOv1().to(device)
loss_fn = SSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

for i in range(10):
    model.train()
    train_loss = 0

    for data, label in train_data:
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predict = model(data)

        loss = loss_fn(predict, label)
        loss.backward()
        optimizer.step()

        train_loss = loss.item() / len(train_data)

    if not i % 5:
        print(train_loss)
