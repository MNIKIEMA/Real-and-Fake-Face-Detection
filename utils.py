import torch
import os
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

# Functon to plot story
# function to plot metrics
# Function to plot images and labels
class FaceDataset(Dataset):

    def __init__(self, image_dir):
        
        """Function to load images into Tensor
            Args: 
                - image_dir : directory of images
                - Return : a dictonary with images and labels
                """
        self.image_dir = image_dir
        self.image_dict = self.load_image()

    def __len__(self) :
        return len(self.image_dict["label"])


    def __getitem__(self, index) :
        img = torch.from_numpy(io.imread(self.image_dict["img_dir"][index],
        as_gray=True,plugin='matplotlib').astype(float))
        labels = torch.Tensor([(l=="real") for l in self.image_dict["label"][index]], dtype=torch.FloatTensor)
        return img, labels


    def load_image(self) :
        img_dict = {"img_dir" : [], "label" : []}
        for root, dirs, files in os.walk(self.image_dir):
            for img in files:
                img_dict["img_dir"].append(os.path.join(root, img))
                img_dict["label"].append(img[:4])
                #print(f"{os.path.join(root, img)} ===>{img[:4]}")
        return img_dict


def plot_history(history, figsize=(8,6), 
                 plot={"Accuracy":['accuracy','val_accuracy'], 'Loss':['loss', 'val_loss']},
                 save_as='auto'):
    """
    Show history
    args:
        history: history
        figsize: fig size
        plot: list of data to plot : {<title>:[<metrics>,...], ...}
    """
    
    for title,curves in plot.items():
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        for c in curves:
            plt.plot(history.history[c])
        plt.legend(curves, loc='upper left')
        plt.show()

def plot_images():
    pass