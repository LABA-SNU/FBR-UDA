from datasets.dataset import CustomDataset
from utils.transforms_utils import transform, augmentation, augmentation_weak, augmentation_strong
import pickle

def get_dataset(crop, dataset_opts, aug_type=None):

    crop = crop
    type = dataset_opts['type']
    print(type)
    
    try: 
        transform = dataset_opts['transform']
    except:
        transform = None

    try:
        augmentation = dataset_opts['augmentation']
    except:
        augmentation = None
    
    if crop in ['apple']:
        
        with open(f'data/{crop}/exp_structure.pickle', 'rb') as fr:
            data_structure = pickle.load(fr)  

        if type in ['src_bg_augmented', 'src_augmented']: 
            
            image_root = '/disks/ssd1/jwoosang1/plant_disease/apple/PV/bg_composed'
            labels = '/disks/ssd1/jwoosang1/plant_disease/apple/PV/pv_labels.pickle'
            flag = 'labeled'
            img_names = data_structure['source']
            
        elif type == 'src_lab': 

            image_root = '/disks/ssd1/jwoosang1/plant_disease/apple/PV/images'
            labels = '/disks/ssd1/jwoosang1/plant_disease/apple/PV/pv_labels.pickle'
            flag = 'labeled'
            img_names = data_structure['source']
        
        elif type == 'src_real': 

            image_root = '/disks/ssd1/jwoosang1/plant_disease/apple/plantpathology/images'
            labels = '/disks/ssd1/jwoosang1/plant_disease/apple/plantpathology/apple_labels.pickle'
            flag = 'labeled'
            img_names = data_structure['source']
            
        elif type == 'tgt': 

            image_root = '/disks/ssd1/jwoosang1/plant_disease/apple/plantpathology/images'
            labels = '/disks/ssd1/jwoosang1/plant_disease/apple/plantpathology/apple_labels.pickle'
            flag = 'unlabeled'
            img_names = data_structure['target']
            
        elif type == 'tst': 
            
            image_root = '/disks/ssd1/jwoosang1/plant_disease/apple/plantpathology/images'
            labels = '/disks/ssd1/jwoosang1/plant_disease/apple/plantpathology/apple_labels.pickle'
            flag = 'unlabeled'
            img_names = data_structure['test']
        
        
        return CustomDataset(img_folder = image_root, 
                             json_pickle = labels, 
                             flag = flag,
                             img_names = img_names,
                             transform = transform,
                             augmentation = augmentation,
                            )
    else:
        raise RuntimeError("Dataset for {} not available".format(crop))
