from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import SensorDataset
import logging
import time # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def inference(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model):

    device = torch.device(device)
    
    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(path_to_save_model+best_model))
    criterion = torch.nn.MSELoss()

    val_loss = 0
    with torch.no_grad():

        model.eval()
        for plot in range(25):

            for index_in, index_tar, _input, target in dataloader:
                
                # starting from 1 so that src matches with target, but has same length as when training
                src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
                target = target.permute(1,0,2).double().to(device) # t48 - t59

                next_input_model = src
                all_predictions = []

                for i in range(forecast_window - 1):
                    
                    prediction = model(next_input_model, device) # 47,1,1: t2' - t48'

                    if all_predictions == []:
                        all_predictions = prediction # 47,1,1: t2' - t48'
                    else:
                        all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

                    pos_encoding_old_vals = src[i+1:, :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                    pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
                    pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48         
                    next_input_model = torch.cat((src[i+1:, :, 0], prediction[-1,:,0].unsqueeze(0))) #t2 -- t47, t48'
                    #print(pos_encodings)
                    #print(next_input_model.unsqueeze(-1))
                    next_input_model = torch.cat((next_input_model.unsqueeze(-1), pos_encodings), dim = 2) # 47, 1, 7 input for next round

                true = torch.cat((src[1:,:,0],target[:-1,:,0]))
                loss = criterion(true, all_predictions[:,:,0])
                val_loss += loss

            val_loss = val_loss/10
            scaler = load('scalar_item.joblib')
            src_x = scaler.inverse_transform(src[:,:,0].cpu())
            src_y = scaler.inverse_transform(src[:,:,1].cpu())
            target_x = scaler.inverse_transform(target[:,:,0].cpu())
            target_y = scaler.inverse_transform(target[:,:,1].cpu())
            prediction_x = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())
            prediction_y = scaler.inverse_transform(all_predictions[:,:,1].detach().cpu().numpy())
            plot_prediction(plot, path_to_save_predictions, src_x, target_x, prediction_x, src_y, target_y, prediction_y, index_in, index_tar)

        logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")

        diference_x = all_predictions[0, 0, 0] - src[0, 0, 0]
        diference_y = all_predictions[0, 0, 1] - src[0, 0, 1]

        if diference_x > 0.5:
            print("??????????????????")
        elif diference_x < -0.5:
            print("??????????????????")

        if diference_y > 0.5:
            print("??????????????????")
        elif diference_y < -0.5:
            print("??????????????????")

        if diference_x < 0.5 and diference_x > -0.5 and diference_y < 0.5 and diference_y > -0.5:
            print("??????")
