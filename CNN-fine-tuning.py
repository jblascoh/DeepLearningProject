import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import trange
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import timm

#Elegir modelo preentrenado
backbone = timm.create_model('mobilenetv3_small_050.lamb_in1k',pretrained=True, features_only=True, out_indices=[-1])
backbone = backbone.eval()
data_config = timm.data.resolve_model_data_config(backbone)
transforms_test = timm.data.create_transform(**data_config, is_training=False)

#Congelar backbone
class CustomModel(nn.Module):
     def __init__(self, backbone, num_classes):
         super().__init__()
         self.backbone = backbone.eval()
         self.classifier_layers = nn.Sequential(
              nn.Conv2d(in_channels=288, out_channels= 512 ,kernel_size= (3,3),stride= (1,1), bias=True,padding='same'),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
              nn.Dropout(0.2),
              nn.Conv2d (in_channels=512, out_channels=1024, kernel_size=(3,3), stride=(1,1), bias=True, padding='same'),
              nn.ReLU(inplace=True ) ,
              nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
              nn.Dropout(0.2),
              nn.Flatten(),
              nn.Linear(1024, num_classes)
         )

     def forward(self, x):
         features = self.backbone.forward(x)[0]
         return self.classifier_layers(features)

     def predict(self,x):
        logits = self.forward(x)
        return(F.softmax(logits))

#Adaptar modelo
num_classes = 6
model = CustomModel(backbone=backbone,  num_classes=num_classes)

for param in model.backbone.parameters():
     param.requires_grad = False

learning_rate = 0.001
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

data_config = timm.data.resolve_model_data_config(backbone)
preprocessing_transform = timm.data.create_transform(**data_config, is_training=False)

train_data_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
    transforms.RandomHorizontalFlip(),
])

train_transform = transforms.Compose([
   train_data_augmentation,
   transforms_test
])

test_transform=transforms_test

def sorted_directory_listing_with_os_listdir(directory):
    items = os.listdir(directory)
    sorted_items = sorted(items)
    return sorted_items

class FlowFromDirectoryDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dir_list = [dir for dir in sorted_directory_listing_with_os_listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,dir)) ]
        self.n_classes = len(self.dir_list)
        self.file_paths = [os.path.join(root,name)   for root,dir,files in os.walk(dataset_path)  for name in files if name.endswith('.jpg') if os.path.split(root)[-1] in self.dir_list]
        self.label_str_list = [os.path.split(root)[-1]   for root,dir,files in os.walk(dataset_path)  for name in files if name.endswith('.jpg') if os.path.split(root)[-1] in self.dir_list]
        self.labels = [self.dir_list.index(os.path.split(root)[-1])   for root,dir,files in os.walk(dataset_path)  for name in files if name.endswith('.jpg') if os.path.split(root)[-1] in self.dir_list]
        self.transform = transform

    def update_transform(self ,transform=None):
        self.transform = transform

    def get_dir_list(self):
        return self.dir_list

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

dataset_path = 'FotosTrainValidation'
full_dataset = FlowFromDirectoryDataset(dataset_path,train_transform)
full_size = len(full_dataset)
train_size = int(0.8 * full_size)
val_size = full_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
print (len(full_dataset))

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

testset_path = 'FotosTest'
test_dataset = FlowFromDirectoryDataset(testset_path,test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)
print(len(test_dataset))

def inverse_transform_np(img):

    n_mean = np.array([0.485, 0.456, 0.406])
    n_std = np.array([0.229, 0.224, 0.225])

    img = np.transpose(img,(1, 2, 0))
    img = (img * n_std + n_mean).clip(0, 1)

    return img

imgs, labels = next(iter(train_dataloader))
print(imgs.shape, labels.shape)

labels_text_list = ['Brooklyn_Bridge', 'Eiffel_Tower', 'Golden_Gate', 'Great_Wall', 'Sagrada_Familia', 'Tokyo_Tower']

r, c = 3, 4
fig = plt.figure(figsize=(c*2, r*2))
for _r in range(r):
    for _c in range(c):
        ix = _r*c + _c
        ax = plt.subplot(r, c, ix + 1)
        img, label = imgs[ix], labels[ix]
        ax.axis("off")
        ax.imshow(inverse_transform_np(img))
        ax.set_title(f'{labels_text_list[label.item()]}', color="green")
plt.tight_layout()
plt.show()

def fit(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=40):
    model.to(device)

    epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc = [], [], [], []
    for epoch in range(1, epochs+1):
        model.train()
        train_loss, train_acc = [], []
        iterator_train = iter(train_dataloader)
        trange_bar = trange(len(train_dataloader))
        for step in trange_bar:
            X, y = next(iterator_train)
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat=model(X)
            loss=criterion(y_hat,y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            train_acc.append(acc)
            trange_bar.set_description(f"loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")
        val_loss, val_acc = [], []
        model.eval()
        iterator_valid = iter(val_dataloader)
        with torch.no_grad():
            trange_bar = trange(len(val_dataloader))
            for step in trange_bar:
                X, y = next(iterator_valid)
                X, y = X.to(device), y.to(device)
                y_hat=model(X)
                loss=criterion(y_hat,y)
                val_loss.append(loss.item())
                acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
                val_acc.append(acc)
                trange_bar.set_description(f"val_loss {np.mean(val_loss):.5f} val_acc {np.mean(val_acc):.5f}")
        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f} acc {np.mean(train_acc):.5f} val_acc {np.mean(val_acc):.5f}")

        epoch_train_loss.append(np.mean(train_loss))
        epoch_val_loss.append(np.mean(val_loss))
        epoch_train_acc.append(np.mean(train_acc))
        epoch_val_acc.append(np.mean(val_acc))

    return epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc

#Entrenar el modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = torch.nn.CrossEntropyLoss()
train_loss, train_acc, val_loss, val_acc = fit(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=8)

#Guardar pesos
torch.save(model.state_dict(), 'model_frozen_backbone.pth')

#Dibujar curvas loss
plt.figure(figsize=[8,8])
plt.plot(train_loss,'r',linewidth=3.0)
plt.plot(val_loss,'b',linewidth=3.0)
plt.legend(['Training loss','Validation Loss'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.draw()
#Dibujar curvas accuracy
plt.figure(figsize=[8,8])
plt.plot(train_acc,'r',linewidth=3.0)
plt.plot(val_acc,'b',linewidth=3.0)
plt.legend(['Training Accuracy','Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()

#Descongelar backbone
for param in model.backbone.parameters():
    param.requires_grad = True

#Cargar pesos guardados
model.load_state_dict(torch.load('model_frozen_backbone.pth'))

#Ajustar parámetros para el fine-tuning y volver a entrenar
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train_loss, train_acc, val_loss, val_acc = fit(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=2)

def dataset_inference(model, criterion, test_dataloader):
    model.to(device)
    test_loss, test_acc, y_hat_acum = [], [], []
    model.eval()
    iterator_valid = iter(test_dataloader)
    with torch.no_grad():
        trange_bar = trange(len(test_dataloader))
        for step in trange_bar:
            X, y = next(iterator_valid)
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat_acum.append(y_hat)
            loss = criterion(y_hat, y)
            test_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            test_acc.append(acc)
            trange_bar.set_description(f"test_loss {np.mean(test_loss):.5f} test_acc {np.mean(test_acc):.5f}")
    return y_hat_acum


# Evaluar modelo con test set
y_hat_acum = dataset_inference(model, criterion, test_dataloader)
y_test_hat = torch.cat(y_hat_acum)
y_test_hat_index = np.argmax(y_test_hat.cpu().numpy(),axis=1)

classes = [0,1,2,3,4,5]

img_test_complete, y_test_labels_complete = next(iter(test_dataloader))

img_test_complete = img_test_complete.cpu().numpy()
y_test_labels_complete = y_test_labels_complete.cpu().numpy()

cm = confusion_matrix(y_test_labels_complete, y_test_hat_index, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
disp.plot()
plt.show()

#Métricas
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP

accuracy = accuracy_score(y_test_labels_complete, y_test_hat_index)
precision = precision_score(y_test_labels_complete, y_test_hat_index, average='macro')
recall = recall_score(y_test_labels_complete, y_test_hat_index, average='macro')
f1 = f1_score(y_test_labels_complete, y_test_hat_index, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

r, c = 12, 11
fig = plt.figure(figsize=(c*2, r*2))
for _r in range(r):
    for _c in range(c):
        ix = _r*c + _c
        ax = plt.subplot(r, c, ix + 1)
        img, label, pred = img_test_complete[ix], y_test_labels_complete[ix], y_test_hat_index[ix]
        ax.axis("off")
        ax.imshow(inverse_transform_np(img))
        ax.set_title(f'{labels_text_list[label.item()]}/{labels_text_list[pred.item()]}', color="green" if label.item() == pred.item() else 'red')
plt.tight_layout()
plt.show()