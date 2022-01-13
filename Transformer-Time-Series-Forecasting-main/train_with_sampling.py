from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import SensorDataset
import logging
import time  # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math, random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                    datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def flip_from_probability(p):
    return True if random.random() < p else False


def transformer(dataloader, EPOCH, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions,
                device):
    device = torch.device(device)

    model = Transformer().double().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss_x = float('inf')
    min_train_loss_y = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss_x = 0
        train_loss_y = 0
        val_loss = 0

        ## TRAIN -- TEACHER FORCING
        model.train()
        for index_in, index_tar, _input, target in dataloader:

            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]

            optimizer.zero_grad()
            src = _input.permute(1, 0, 2).double().to(device)[:-1, :, :]
            target = _input.permute(1, 0, 2).double().to(device)[1:, :, :]
            sampled_src_x = src[:1, :, :]
            sampled_src_y = src[1:2, :, :]

            for i in range(len(target) - 1):

                prediction_x = model(sampled_src_x, device)[:, :, :-1]
                prediction_y = model(sampled_src_y, device)[:, :, 1:2]

                # for p1, p2 in zip(params, model.parameters()):
                #     if p1.data.ne(p2.data).sum() > 0:
                #         ic(False)
                # ic(True)
                # ic(i, sampled_src[:,:,0], prediction)
                # time.sleep(1)
                """
                # to update model at every step
                # loss = criterion(prediction, target[:i+1,:,:1])
                # loss.backward()
                # optimizer.step()
                """

                if i < 24:  # One day, enough data to make inferences about cycles
                    prob_true_val = True
                else:
                    ## coin flip
                    v = k / (k + math.exp(epoch / k))  # probability of heads/tails depends on the epoch, evolves with time.
                    prob_true_val = flip_from_probability(v)  # starts with over 95 % probability of true val for each flip in epoch 0.
                    ## if using true value as new value

                if prob_true_val:  # Using true value as next value
                    sampled_src_x = torch.cat((sampled_src_x.detach(), src[i + 1, :, :].unsqueeze(0).detach()))
                    sampled_src_y = torch.cat((sampled_src_y.detach(), src[i + 1, :, :].unsqueeze(0).detach()))
                else:  ## using prediction as new value
                    positional_encodings_new_val = src[i + 1, :, 1:].unsqueeze(0)
                    predicted_x = torch.cat((prediction_x[-1, :, :].unsqueeze(0), positional_encodings_new_val), dim=2)
                    predicted_y = torch.cat((prediction_y[-1, :, :].unsqueeze(0), positional_encodings_new_val), dim=2)
                    sampled_src_x = torch.cat((sampled_src_x.detach(), predicted_x.detach()))
                    sampled_src_y = torch.cat((sampled_src_y.detach(), predicted_y.detach()))

            """To update model after each sequence"""
            loss_x = criterion(target[:-1, :, 0].unsqueeze(-1), prediction_x)
            loss_y = criterion(target[:-1, :, 1].unsqueeze(-1), prediction_y)
            loss_x.backward()
            loss_y.backward()
            optimizer.step()
            train_loss_x += loss_x.detach().item()
            train_loss_y += loss_y.detach().item()

        if train_loss_x < min_train_loss_x or train_loss_y < min_train_loss_y:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            best_model = f"best_train_{epoch}.pth"
            if train_loss_x < min_train_loss_x:
                min_train_loss_x = train_loss_x
            if train_loss_y < min_train_loss_y:
                min_train_loss_y = train_loss_y

        if epoch % 10 == 0:
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss_x, train_loss_y}")
            scaler = load('scalar_item.joblib')
            sampled_src_x = scaler.inverse_transform(sampled_src_x[:, :, 0].cpu())
            sampled_src_y = scaler.inverse_transform(sampled_src_y[:, :, 1].cpu())
            src_x = scaler.inverse_transform(src[:, :, 0].cpu())  # torch.Size([35, 1, 7])
            src_y = scaler.inverse_transform(src[:, :, 1].cpu())
            target_x = scaler.inverse_transform(target[:, :, 0].cpu())  # torch.Size([35, 1, 7])
            target_y = scaler.inverse_transform(target[:, :, 1].cpu())  # torch.Size([35, 1, 7])
            prediction_x = scaler.inverse_transform(
                prediction_x[:, :, 0].detach().cpu().numpy())  # torch.Size([35, 1, 7])
            prediction_y = scaler.inverse_transform(prediction_y[:, :, 0].detach().cpu().numpy())
            plot_training_3(epoch, path_to_save_predictions, src_x, sampled_src_x, prediction_x, src_y, sampled_src_y,
                            prediction_y)

        train_loss_x /= len(dataloader)
        train_loss_y /= len(dataloader)
        log_loss(train_loss_x, train_loss_y, path_to_save_loss, train=True)

    plot_loss(path_to_save_loss, train=True)
    return best_model
