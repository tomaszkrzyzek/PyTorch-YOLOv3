import torch
import torch.nn
import torch.utils.data
import torch.hub
import torch.optim
import torchvision.transforms
import numpy as np
import os

from tqdm import tqdm

import utils.datasets
# from paths import *

device = 'cpu'


# def collect_paths_to_image(path):
#     paths_list = []
#     for content in tqdm(os.walk(path), desc='Read dataset:'):
#         if content[2] and content[0][-6:] == 'images':
#             path_to_image = content[0] + '/' + content[2][0]
#             paths_list.append(path_to_image)
#     return paths_list


# def collect_paths_to_masks(path):
#     paths_list = []
#     for content in tqdm(os.walk(path), desc='Read dataset:'):
#         if content[2] and content[0][-5:] == 'masks':
#             path_to_mask = content[0] + '/' + content[2][0]
#             paths_list.append(path_to_mask)
#     return paths_list


class YOLO(torch.nn.Module):

    def __init__(self, bounding_box_shape):
        super(YOLO, self).__init__()
        output_size = bounding_box_shape[0] * bounding_box_shape[1]
        self.convolution = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True)
        self.bounding_box = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=1000, out_features=4000),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=4000, out_features=4000),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=4000, out_features=output_size),
        )
        self.bounding_box_shape = bounding_box_shape

    def forward(self, x):
        x = self.convolution(x)
        x = self.bounding_box(x)
        x = torch.reshape(x, shape=self.bounding_box_shape)
        return x


preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
PATH_TO_TEST = '../data-science-bowl-2018/stage1_test'
PATH_TO_TRAIN = '../data-science-bowl-2018/stage1_train'
PATH_TO_TEST_FINAL = '../data-science-bowl-2018/stage2_test_final'
# paths_to_images_train = collect_paths_to_image(PATH_TO_TRAIN)
# paths_to_masks_train = collect_paths_to_masks(PATH_TO_TRAIN)
# train = utils.datasets.ListDataset(img_paths=paths_to_images_train, label_paths=paths_to_masks_train)
train = utils.datasets.PandasDataset(PATH_TO_TRAIN, augment=True, multiscale=True,normalized_labels=False)
# paths_to_images_test = collect_paths_to_image(PATH_TO_TEST)
# paths_to_masks_test = collect_paths_to_masks(PATH_TO_TEST)
# test = utils.datasets.ListDataset(img_paths=paths_to_images_test, label_paths=paths_to_masks_test)
test = utils.datasets.PandasDataset(PATH_TO_TEST, augment=False, multiscale=False)

batch_size = 128
learning_rate = 0.0001
epoch = 30

bounding_box_shape = (3, 3, 7)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

model = YOLO(bounding_box_shape).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss()

for epoch_number in range(epoch):
    print("Epoch: ", epoch_number)

    model.train()
    for x, y in tqdm(train_loader, desc="Training: "):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss_value = loss_function(output, y)
        loss_value.backward()
        optimizer.step()

    model.eval()

    test_loss = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Evaluate: '):
            x, y = x.to(device), y.to(device)
            output = model(x)
            test_loss += np.mean(loss_function(output, y).item())

        print('\nLoss value: {}'.format(test_loss))
        


