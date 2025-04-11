from custom_dataset import MyCustomDataset
from torch.utils.data import DataLoader

def get_dataloader(name):
    if name == 'xmedianet_deep':
        config = {
        'dataset_name': 'xmedianet_deep',
        'class_number': 200,
        'input_dim_I': 4096,  
        'input_dim_T': 300,
        'threshold': 0.5,
        'batch_size': 100,
        'alpha': 1,
        'layer_num': 3
        }
    elif name == 'INRIA':
        config = {'dataset_name': 'INRIA',
        'class_number': 100,
        'input_dim_I': 4096, 
        'input_dim_T': 1000,
        'threshold': 0.5,
        'batch_size': 300,
        'alpha': 0.05,
        'lr':5e-5,
        'layer_num': 4
        }
    elif name == 'nus_deep':
        config = {
        'dataset_name': 'nus_deep',
        'class_number': 10,
        'input_dim_I': 4096, 
        'input_dim_T': 300,
        'threshold': 0.5,
        'batch_size': 300,
        'alpha': 0.5,
        'lr': 5e-5, 
        'layer_num': 4
        }
    elif name == 'wiki':
        config = {
        'dataset_name': 'wiki',
        'class_number': 10,
        'input_dim_I': 4096, 
        'input_dim_T': 300,
        'threshold': 0.2,
        'batch_size': 120, 
        'alpha': 0.05,
        'lr': 5e-5, 
        'layer_num': 3
        }
    elif name == 'pascal':
        config = {
        'dataset_name': 'pascal',
        'class_number': 20,
        'input_dim_I': 4096, 
        'input_dim_T': 300,
        'threshold': 0.71,
        'batch_size': 100,
        'alpha': 10,
        'lr':5e-5,
        'layer_num': 3
        }

    dataset = {x: MyCustomDataset(dataset=config['dataset_name'], state=x)
                    for x in ['train', 'val','test']}
    data_loaders = {x: DataLoader(dataset[x], batch_size=config['batch_size'], num_workers=1)
                        for x in ['train','val', 'test']}
    
    print('-' * 80)
    for key in config:
        print(key, ": ", config[key])
    return config, data_loaders