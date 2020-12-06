
import os
import numpy as np
import torch

class DuckietownSegmentationDataset(object):
    def __init__(self, root, transforms):  # would be 'data_collection/
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "data_collection/dataset"))))


    def __getitem__(self, idx):
        # load images ad masks
        dataframe_path = os.path.join(self.root, "data_collection/dataset", self.imgs[idx])
        dataframe = np.load(dataframe_path)

        img = dataframe["arr_0"]     # image is a numpy array, in rgb order
        boxes = dataframe["arr_1"]
        labels = dataframe["arr_2"]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



from model import Model
from engine import train_one_epoch
import utils

def main():
    # TODO train loop here!

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    root = os.path.dirname(os.path.dirname(__file__))
    dataset = DuckietownSegmentationDataset(root, get_transform(train=True))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
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

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

    save_model_path = 'weights/model.pt'
    print('saving trained model to {}'.format(save_model_path))
    model.save_model(save_model_path)

    # TODO don't forget to save the model's weights inside of `./weights`!


if __name__ == "__main__":
    main()