import torch
import numpy as np
import random
import os
from options import get_dataloader
from model import FUME
from processor import train_val_test_model
import time

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    set_seed(2)
    datasetName = 'pascal' # 'pascal', 'wiki', 'nus_deep', 'INRIA', 'xmedianet_deep'

    # Construct folder
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    folder_path = os.path.join("saved", str(datasetName), cur_time)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    dataset_config, data_loaders = get_dataloader(datasetName)
    
    hyperparameter = {
        'MAX_EPOCH': 100, 
        'betas':(0.5, 0.999),
        'weight_decay':0,
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }  

    for key in hyperparameter:
        print(key, ": ", hyperparameter[key])
    print('-' * 80)
    
    model = FUME(hyperparameter['device'],
                    img_input_dim=dataset_config['input_dim_I'], 
                    text_input_dim=dataset_config['input_dim_T'], 
                    output_dim=dataset_config['class_number'],
                    num_class=dataset_config['class_number'],
                    layer_num=dataset_config['layer_num']).to(hyperparameter['device'])

    print("model:" , model)
    params_to_update = list(model.parameters())

    optimizer = torch.optim.Adam(params_to_update, 
                           lr=dataset_config['lr'], 
                           betas=hyperparameter['betas'])

    print('...Training is beginning...')

    # Train validate and test FUME
    results = train_val_test_model(model, 
                                data_loaders, 
                                optimizer, 
                                dataset_config['alpha'], 
                                hyperparameter['MAX_EPOCH'], 
                                hyperparameter['device'],
                                folder_path)
    
    print('...Training is completed...')
    print('#' * 80)
    for key in results:
        print(key, ": ", round(results[key], 3))
    print('#' * 80)

