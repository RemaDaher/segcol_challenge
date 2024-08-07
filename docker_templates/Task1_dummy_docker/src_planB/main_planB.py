"""
SegcCol challenge - MICCAI 2024
Challenge link: 

This is a dummy example to illustrate how participants should format their prediction outputs.
"""

import numpy as np
import glob
from PIL import Image
from torchvision import transforms
import os
from skimage.util import random_noise

os.chdir('/home/rema/workspace/segcol_challenge/docker_templates/Task1_dummy_docker/src')
import sys
sys.path.append('/home/rema/workspace/segcol_challenge/docker_templates/Task1_dummy_docker/src/Folds_Baseline_Dexined')

from Folds_Baseline_Dexined.main import main, parse_args



to_tensor = transforms.ToTensor()

def get_images(file):
    # Open the image
    img = Image.open(file)

    # Convert the image data to a numpy array
    img_data = np.array(img)


    # Create class channels using np.where
    class1 = np.where(img_data == 255, 1, 0) # fold
    class2 = np.where(img_data == 127, 1, 0) # tool1 
    class3 = np.where(img_data == 128, 1, 0) # tool2
    class4 = np.where(img_data == 129, 1, 0) # tool3

    # Stack all class channels together
    img_data = np.stack((class1, class2, class3, class4), axis=-1)

    return img_data

def predict_masks(pred_folder, output_folder):
    """
    param im_path: Input path for a single image
    param output_folder: Path to folder where output will be saved
    predict the segmentation masks for an image and save in the correct formatting as .npy file
    """
    ### replace below with your own prediction pipeline ###
    # We generate dummy predictions here with noise added to ground truth
    
    # pred_orig = get_images(pred_file).astype(np.float64) #dummy make sure pred is float

    # prediction = random_noise(pred_orig, mode='gaussian', var=0.1, rng = 42)#.astype(np.float16) #dummy
    args = parse_args()
    args.checkpoint_data= '16/16_model.pth'
    args.test_data = 'CLASSIC'
    args.test_img_width = 640
    args.test_img_height = 480
    args.input_val_dir = pred_folder
    args.output_dir = '/home/rema/workspace/segcol_challenge/docker_templates/Task1_dummy_docker/src/Folds_Baseline_Dexined/checkpoints' # path to checkpoint_data
    args.train_data = '' # can keep default
                     
    fuse, _ = main(args)



    ### End of prediction pipeline ###

    # Save the predictions in the correct format

    
    pred_files = np.sort(glob.glob(f'{pred_folder}/*'))
    for i in range(len(pred_files)):
        prediction_fold = (1-(fuse[i]/255))
        prediction = np.expand_dims(prediction_fold, -1).repeat(4, axis =-1).astype(np.float64)
        # Change the format of predction to nparray and save
        assert type(prediction[i]) == np.ndarray, \
            "Wrong type of predicted depth, expected np.ndarray, got {}".format(type(prediction))
        
        assert prediction.shape == (480, 640, 4), \
            "Wrong size of predicted depth, expected (480, 640, 4), got {}".format(list(prediction.shape))
        
        assert prediction.dtype == np.float64, \
        "Wrong data type of predicted depth, expected np.float64, got {}".format(prediction.dtype)
        output_file =  pred_files[i].replace(input_path, output_path).replace('imgs','predictions').replace('.png','.npy')  #main_path2
        output_folder = os.path.dirname(output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(output_folder +' created')
        np.save(output_file, prediction)
        print(f"Saved prediction to {output_file}")


if __name__ == "__main__":
    input_path = "/home/rema/workspace/segcol_challenge/data/input"#sys.argv[1]
    output_path = "/home/rema/workspace/segcol_challenge/data/output"#sys.argv[2]

    # Replace 'segm_maps' with 'imgs' in line 71 and 75 if you are not using the dummy data
    if not os.path.exists(input_path):
        print('No input folder found')
        exit(1)
    else:
        print(input_path +' exists')
        input_folders = np.sort(glob.glob(f"{input_path}/Seq*"))

    for input_folder in input_folders:
        predict_masks(f'{input_folder}/imgs/', output_path)
