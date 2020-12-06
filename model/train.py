import os
import cv2
import torch
import torchvision
import numpy as np
import transforms as T
import utils

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch
from model import Model, Wrapper


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class DuckDataset(object):
    def __init__(self, dataset_dir, transforms):
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        
        # load all image files, sorting them to
        # ensure that they are aligned
        self.npzs = list(sorted(os.listdir(dataset_dir)))

        # remove the readme file
        if "README.md" in self.npzs:
            self.npzs.remove("README.md")

    def __getitem__(self, idx):
        # load npz files
        npz_path = os.path.join(self.dataset_dir, self.npzs[idx])
        npz = np.load(npz_path)

        img = npz[f"arr_{0}"] # Check if conversion to rgb is needed
        boxes = npz[f"arr_{1}"]
        classes = npz[f"arr_{2}"]

        # convert boxes and labels into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32) #maybe for img too
        labels = torch.as_tensor(classes, dtype=torch.int64)
        image_id = torch.as_tensor([int(self.npzs[idx].split(".")[0])])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # img = torch.reshape(img, (224, 224, 3))

        return img, target

    def __len__(self):
        return len(self.npzs)


def main():
    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of `./weights`!

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create the dataset object
    dataset_dir = "../data_collection/dataset"
    dataset = DuckDataset(dataset_dir, get_transform(train=True))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)


    model = Model()
    
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    print("Starting training!")
    print(f"Dataset size: {len(dataset)}")

    num_epochs = 4

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=40)
        # update the learning rate
        lr_scheduler.step()

    print("That's it for the training part!")

    # Save the model weights
    torch.save(model.state_dict(), "./weights/model.pt")


if __name__ == "__main__":
    main()