import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.U2Net import U2NET
from dataset import PPGECGDataset
from scipy.stats import pearsonr
import pickle
import matplotlib.pyplot as plt
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR

def he_initialization(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

mse_loss = torch.nn.MSELoss()

def muti_mse_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = mse_loss(d0,labels_v)
	loss1 = mse_loss(d1,labels_v)
	loss2 = mse_loss(d2,labels_v)
	loss3 = mse_loss(d3,labels_v)
	loss4 = mse_loss(d4,labels_v)
	loss5 = mse_loss(d5,labels_v)
	loss6 = mse_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss

def train_model(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        d0, d1, d2, d3, d4, d5, d6 = model(inputs)
        loss = muti_mse_loss_fusion(d0, d1, d2, d3, d4, d5, d6, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def predict_model(model, dataloader, device):
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            d0, d1, d2, d3, d4, d5, d6= model(inputs)
            predictions.append(d0)
            ground_truths.append(targets)

    return predictions, ground_truths

def calculate_pearson_coefficient(predictions, ground_truths):
    preds = torch.cat(predictions).flatten().cpu().numpy()
    gt = torch.cat(ground_truths).flatten().cpu().numpy()
    coeff, _ = pearsonr(preds, gt)
    return coeff

def sort_files_by_number(file_list):
    return sorted(file_list, key=lambda x: [int(s) for s in re.findall(r'\d+', x)])

train_data_folder = r"I:\sci\code\code\mimic\processed_train"
test_data_folder = r"I:\sci\code\code\mimic\processed_predict"
predict_data_folder = "subject_depend_predict_data_u2net"

train_loss_folder = "trainloss_u2net"
coeff_folder = "coeff_u2net"

os.makedirs(train_loss_folder, exist_ok=True)
os.makedirs(coeff_folder, exist_ok=True)

if not os.path.exists(predict_data_folder):
    os.makedirs(predict_data_folder)

train_file_list = sort_files_by_number([f for f in os.listdir(train_data_folder) if os.path.splitext(f)[1] == '.p'])
test_file_list = sort_files_by_number([f for f in os.listdir(test_data_folder) if os.path.splitext(f)[1] == '.p'])
log_dir = "tensorboard_logs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = []
pearson_coefficients = []

for train_idx, (train_file, test_file) in enumerate(zip(train_file_list[:], test_file_list[:])):
    train_dataset = PPGECGDataset(os.path.join(train_data_folder, train_file))
    test_dataset = PPGECGDataset(os.path.join(test_data_folder, test_file))

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize the model, loss function, and optimizer for each iteration
    model = U2NET()
    he_initialization(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    epoch_train_losses = []
    epoch_pearson_coefficients = []


    predictions = None  # Initialize the 'predictions' variable
    ground_truths = None  # Initialize the 'ground_truths' variable

    for epoch in tqdm(range(200), desc=f"Training {train_file}"):
        train_loss = train_model(model, train_dataloader, optimizer, device)
        epoch_train_losses.append(train_loss)


        # Record Pearson coefficient for test dataset
        predictions, ground_truths = predict_model(model, test_dataloader, device)
        coeff = calculate_pearson_coefficient(predictions, ground_truths)
        epoch_pearson_coefficients.append(coeff)

    train_losses.append(epoch_train_losses)
    pearson_coefficients.append(epoch_pearson_coefficients)

    if predictions is not None and ground_truths is not None:  # Save predictions and ground_truths only if they are defined
        with open(os.path.join(predict_data_folder, f"predictions_and_ground_truth_{test_file}"), "wb") as f:
            pickle.dump((predictions, ground_truths), f)

    # Plot the training loss
    plt.figure()
    plt.plot(epoch_train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss - {train_file}')
    plt.savefig(f'trainloss_u2net/training_loss_{train_file}.png')  # Save the plot as an image
    plt.show()  # Display the plot


    # Plot the Pearson coefficients
    plt.figure()
    plt.plot(epoch_pearson_coefficients)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Coefficient')
    plt.title(f'Pearson Coefficient for Test Dataset - {test_file}')
    plt.savefig(f'coeff_u2net/pearson_coefficients_{test_file}.png')  # Save the plot as an image
    plt.show()  # Display the plot


