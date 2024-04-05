import sys

sys.path.append('/home/enigma/3_Liwen/chatter/UnsupervisedCluster/vgg/dataloaders')
sys.path.append('/home/enigma/3_Liwen/chatter/UnsupervisedCluster/vgg/arch')

import argparse
from dataset import SurfaceDataset
from alldata import alldata
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from vgg import VGG16Autoencoder
from vgg_500 import VGG16_500
import wandb
from pytorch_msssim import SSIM
import random
import numpy as np
import os

def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(epochs, learnrate, lossfunc, a, b, savename, alltrain, trainmodel):
    seed_everything(seed=7)
    wandb.init(project='Chatter_Cluster', entity='liwenhu3', name='VGG16 SSIM')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    if alltrain == True:
        
        train_dataset = alldata()
        train_loader = DataLoader(train_dataset, num_workers = 4, batch_size = 64, shuffle = True)
        
    elif alltrain == False:
        train_dataset = SurfaceDataset(train=True, transform=False)
        test_dataset = SurfaceDataset(train=False, transform=False)
    
        train_loader = DataLoader(train_dataset, num_workers=4, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, num_workers=4, batch_size=64, shuffle=True)

    if trainmodel == 'vgg16':
        model = VGG16Autoencoder().to(device)
        model.train()
    elif trainmodel == 'vgg100':
        model = VGG16_500().to(device)
        model.train()
    
    if torch.cuda.device_count() > 1:
        device_ids = [0, 1]
        print(f"Using {len(device_ids)} GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learnrate)
    criterion1 = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    criterion3 = SSIM(data_range=1.0, size_average=True, channel=4)

    wandb.config.update({
        "learning_rate": 5e-4,
        "architecture": "VGG16Autoencoder",
        "dataset": "SurfaceDataset",
        "epochs": epochs,
        "batch_size": 64
    })

    log_interval = 10

    for epoch in range(epochs):
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            if lossfunc == 'ssim':
                loss = 1 - criterion3(outputs, inputs)
            elif lossfunc == 'combined':
                loss = (1 - criterion3(outputs, inputs)) + a * criterion1(outputs, inputs) + b * criterion2(outputs, inputs)
            elif lossfunc == 'l1l2':
                loss = criterion1(outputs, inputs) + criterion2(outputs, inputs)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        if alltrain == True:
            if (epoch + 1) % log_interval == 0:
                # Visualization of 5 input/output examples after training loop
                with torch.no_grad():
                    # Fetch a single batch from the train_loader
                    for sample_inputs, _ in train_loader:
                        sample_inputs = sample_inputs.to(device)
                        sample_outputs = model(sample_inputs)
                        break  # Only take the first batch
        
                    # Select 5 samples for visualization
                    selected_inputs = sample_inputs[:5]
                    selected_outputs = sample_outputs[:5]
        
                    input_height_channel = selected_inputs[:, 3, :, :].cpu()
                    input_optical_channel = selected_inputs[:, 0:3, :, :].cpu()
        
                    output_height_channel = selected_outputs[:, 3, :, :].cpu()
                    output_optical_channel = selected_outputs[:, 0:3, :, :].cpu()
        
                    wandb.log({
                        "Input Height (Train)": [wandb.Image(input_height_channel[j].unsqueeze(0), caption="Input Height (Train)") for j in range(input_height_channel.shape[0])],
                        "Output Height (Train)": [wandb.Image(output_height_channel[j].unsqueeze(0), caption="Output Height (Train)") for j in range(output_height_channel.shape[0])],
                        "Input Optical (Train)": [wandb.Image(input_optical_channel[j], caption="Input RGB (Train)") for j in range(input_optical_channel.shape[0])],
                        "Output Optical (Train)": [wandb.Image(output_optical_channel[j], caption="Output RGB (Train)") for j in range(output_optical_channel.shape[0])]
                    }, step=epoch)
            wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss},step = epoch)
        
        elif alltrain == False:
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    test_inputs, _ = data
                    test_inputs = test_inputs.to(device)
                    test_outputs = model(test_inputs)
                    if lossfunc == 'ssim':
                        loss = 1 - criterion3(outputs, inputs)
                    elif lossfunc == 'combined':
                        loss = (1 - criterion3(outputs, inputs)) + a * criterion1(outputs, inputs) + b * criterion2(outputs, inputs)
                    elif lossfunc == 'l1l2':
                        loss = criterion1(outputs, inputs) + criterion2(outputs, inputs)
                    total_test_loss += loss.item()
    
            avg_test_loss = total_test_loss / len(test_loader)

            # At the end of the testing loop within each epoch:
            if (epoch + 1) % log_interval == 0:
                with torch.no_grad():
                    # Fetch a single batch from the test_loader
                    for sample_inputs, _ in test_loader:
                        sample_inputs = sample_inputs.to(device)
                        sample_outputs = model(sample_inputs)
                        break  # Only take the first batch
            
                    # Visualization and logging code, modified to log for 2 samples instead of 5
                    selected_inputs = sample_inputs[:5]
                    selected_outputs = sample_outputs[:5]
            
                    input_height_channel = selected_inputs[:, 3, :, :].cpu()
                    input_optical_channel = selected_inputs[:, 0:3, :, :].cpu()
            
                    output_height_channel = selected_outputs[:, 3, :, :].cpu()
                    output_optical_channel = selected_outputs[:, 0:3, :, :].cpu()
            
                    wandb.log({
                        "Input Height (Test)": [wandb.Image(input_height_channel[j].unsqueeze(0), caption="Input Height (Test)") for j in range(input_height_channel.shape[0])],
                        "Output Height (Test)": [wandb.Image(output_height_channel[j].unsqueeze(0), caption="Output Height (Test)") for j in range(output_height_channel.shape[0])],
                        "Input Optical (Test)": [wandb.Image(input_optical_channel[j], caption="Input RGB (Test)") for j in range(input_optical_channel.shape[0])],
                        "Output Optical (Test)": [wandb.Image(output_optical_channel[j], caption="Output RGB (Test)") for j in range(output_optical_channel.shape[0])]
                    },step = epoch)
            
            # Make sure to call wandb.log outside any conditional blocks that might not execute
            wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "test_loss": avg_test_loss},step = epoch)

    torch.save(model.state_dict(), savename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a VGG16 Autoencoder.')
    parser.add_argument('--savename', type=str, default='vgg16_ssim_50.pth', help='Filename for saving the model weights')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--learnrate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--lossfunc', choices=['ssim', 'combined', 'l1l2'], default='ssim', help='Loss function to use: ssim, combined, l1l2')
    parser.add_argument('--a', type=float, default=1.0, help='Weight of L1 loss in combined loss function')
    parser.add_argument('--b', type=float, default=1.0, help='Weight of L2 loss in combined loss function')
    parser.add_argument('--alltrain', type=bool, default=True, help='determine how many data are used: alltrain, train')
    parser.add_argument('--trainmodel', type=str, default='vgg100')
    
    args = parser.parse_args()

    main(args.epochs, args.learnrate, args.lossfunc, args.a, args.b, args.savename, args.alltrain, args.trainmodel)

