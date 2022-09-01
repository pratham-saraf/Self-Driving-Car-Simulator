import os
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from matplotlib import pyplot as plt
import cv2
import matplotlib.image as mpimg
import pandas as pd
# !pip install tensorboard-logger
# from tensorboard_logger import configure, log_value
from torch.utils.data import Dataset
import csv

class CNN_Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout()

        self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2) #384 kernels, size 3x3
        self.conv_3 = nn.Conv2d(48, 64, kernel_size=3) # 384 kernels size 3x3
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # 256 kernels, size 3x3

        self.fc0 = nn.Linear(1152, 100)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, input):
        input = input/127.5-1.0
        input = self.elu(self.conv_0(input))
        input = self.elu(self.conv_1(input))
        input = self.elu(self.conv_2(input))
        input = self.elu(self.conv_3(input))
        input = self.elu(self.conv_4(input))
        input = self.dropout(input)

        input = input.flatten()
        input = self.elu(self.fc0(input))
        input = self.elu(self.fc1(input))
        input = self.elu(self.fc2(input))
        input = self.fc3(input)

        return input

model = CNN_Model()

if torch.cuda.is_available():
        model = model.cuda()

class Features(Dataset):
    def __init__(self, path_to_csv):
        path_to_csv = os.path.join(path_to_csv, 'driving_log.csv')
        print(path_to_csv)
        self.csv_data = self.load_csv_file(path_to_csv)

    def load_csv_file(self, path_to_csv):
        data = []
        with open(path_to_csv, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            next(data_reader)
            for row in data_reader:
                data.append(row)

        return data

    def get_csv_data(self):
        return self.csv_data

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self,i):
        data_entry = self.csv_data[i]

        to_return = {
            'img_center_pth': data_entry[0],
            'img_left_pth': data_entry[1],
            'img_right_pth': data_entry[2],
            'steering_angle': data_entry[3],
            'throttle': data_entry[4],
            'brake': data_entry[5],
            'speed': data_entry[6]
        }

        return to_return

dataset = Features("Data/")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data , val_data = torch.utils.data.dataset.random_split(dataset,[train_size,val_size])

#template used from https://github.com/naokishibuya/car-behavioral-cloning/blob/master/utils.py

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return "img_left_pth", float(steering_angle) + 0.2
    elif choice == 1:
        return "img_right_pth", float(steering_angle) - 0.2
    return "img_center_pth", float(steering_angle)


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

def rectify(data,img_pth):
            # val = data[img_pth].split("\\")[-1]
            val = ("Data/IMG/"+img_pth)
            data[img_pth] = val

def eval_model(model,dataset,num_samples):
    model.eval()
    criterion = nn.MSELoss()
    step = 0
    val_loss = 0
    count = 0
    sampler = RandomSampler(dataset)
    torch.manual_seed(0)
    for sample_id in tqdm(sampler):
        if step==num_samples:
            break

        data = dataset[sample_id]
        img_pth, label = choose_image(data['steering_angle'])
        rectify(data,img_pth)
        
        
        img = cv2.imread(data[img_pth])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess(img)
        img, label = random_flip(img, label)
        img, label = random_translate(img, label, 100, 10)
        img = random_shadow(img)
        img = random_brightness(img)
        img = Variable(torch.cuda.FloatTensor([img]))
        img = img.permute(0,3,1,2)
        label = np.array([label]).astype(float)
        label = Variable(torch.cuda.FloatTensor(label))

        out_vec = model(img)

        loss = criterion(out_vec,label)

        batch_size = 4
        val_loss += loss.data.item()
        count += batch_size
        step += 1

    val_loss = val_loss / float(count)
    return val_loss

model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

model_dir = "models/"
step = 0 
imgs_per_batch = 40
optimizer.zero_grad()
for epoch in range(10):
    sampler = RandomSampler(train_data, replacement=True, num_samples=20000)
    for i, sample_id in enumerate(sampler):
        data = train_data[sample_id]
        
        label = data["steering_angle"]
        img_pth, label = choose_image(label)
#         print("img_pth" ,img_pth)
#         print("data[pth]",data[img_pth])
        rectify(data,img_pth)
#         print("img_pth" ,img_pth)
#         print("data[pth]",data[img_pth])
        img = cv2.imread(data[img_pth])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess(img)
        img, label = random_flip(img, label)
        img, label = random_translate(img, label, 100, 10)
        img = random_shadow(img)
        img = random_brightness(img)
        img = Variable(torch.cuda.FloatTensor([img]))
        label = np.array([label]).astype(float)
        label = Variable(torch.cuda.FloatTensor(label))
        img = img.permute(0,3,1,2)
        
        out_vec = model(img)
        loss = criterion(out_vec,label)

        loss.backward()
        if step%imgs_per_batch==0:
            optimizer.step()
            optimizer.zero_grad()


        if step%100==0:
            log_str = \
                'Epoch: {} | Iter: {} | Step: {} | ' + \
                'Train Loss: {:.8f} |'
            log_str = log_str.format(
                epoch,
                i,
                step,
                loss.item())
            print(log_str)

#         if step%100==0:
#             log_value('train_loss',loss.item(),step)

        if step%5000==0:
            val_loss = eval_model(model,val_data, num_samples=400)
#             log_value('val_loss',val_loss,step)
            log_str = \
                'Epoch: {} | Iter: {} | Step: {} | Val Loss: {:.8f}'
            log_str = log_str.format(
                epoch,
                i,
                step,
                val_loss)
            print(log_str)
            model.train()

        if step%5000==0:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_pth = os.path.join(
                model_dir,
                'model_{}'.format(step))
            torch.save(
                model.state_dict(),
                model_pth)

        step += 1
