import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class Wrapper:
    def __init__(self):
        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU (if you have one)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = Model()
        self.model.load_state_dict(torch.load("weights/model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, batch_or_image, area_thresh=10.0, cone_size_red=0.8, duck_size_red=0.9):
        # TODO: Make your model predict here!

        # The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # batch_size x 224 x 224 x 3 batch of images)
        # These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # etc.

        # This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # second is the corresponding labels, the third is the scores (the probabilities)

        if len(batch_or_image.shape) == 3:
            batch = [batch_or_image]
        else:
            batch = batch_or_image

        with torch.no_grad():
            preds = self.model([to_tensor(cv2.resize(img, (224,224))).to(device=self.device, dtype=torch.float) for img in batch])

        boxes = []
        labels = []
        scores = []
        for pred in preds:
            pred_boxes = pred["boxes"].cpu().numpy()
            
            filt_pred_boxes = []
            filt_pred_labels = []
            filt_pred_scores = []
            
            # Filter predictions that we know to be bad
            for i in range(pred_boxes.shape[0]):
                box = pred_boxes[i]
                label = pred["labels"].cpu().numpy()[i]
                area = (box[2] - box[0]) * (box[3] - box[1])
                
                if area > area_thresh:
                    # alter the box
                    new_box = np.copy(box)

                    if label == 1: # duck
                        new_box[2] = box[0] + duck_size_red * (box[2] - box[0]) # alter the width
                        new_box[1] = box[3] + duck_size_red * (box[1] - box[3]) # alter the height
                    elif label == 2: # cone
                        new_box[2] = box[0] + (cone_size_red + 0.1) * (box[2] - box[0]) # alter the width
                        new_box[1] = box[3] + cone_size_red * (box[1] - box[3]) # alter the height more

                    filt_pred_boxes.append(new_box)
                    filt_pred_labels.append(label)
                    filt_pred_scores.append(pred["scores"].cpu().numpy()[i])

            filt_pred_boxes = np.array(filt_pred_boxes)
            filt_pred_labels = np.array(filt_pred_labels)
            filt_pred_scores = np.array(filt_pred_scores)

            boxes.append(filt_pred_boxes)
            labels.append(filt_pred_labels)
            scores.append(filt_pred_scores)

        return boxes, labels, scores


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # TODO Instantiate your weights etc here!
        # Load the model
        self.model = torchvision.models.detection \
                     .fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

    def forward(self, x, y=None):
        return self.model(x) if y is None else self.model(x, y)
