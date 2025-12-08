from torch.utils.data import DataLoader
import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from utils.custom_transforms import RemoveEyes

from insightface.app import FaceAnalysis


class ScratchDataset():
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = os.listdir(root) # Load your data here
        self.length = len(self.data)    

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.root, self.data[idx]))
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                print(f"Error applying transform to image {self.data[idx]}: {e}")
        return img

def insightface_landmarks_fn(img):
    # If img is a numpy array, skip conversion
    if isinstance(img, np.ndarray):
        img_np = img
    else:
        img_np = np.array(img.convert('RGB'))
    try:
        faces = app.get(img_np)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
        lms = face.landmark.astype(int)
        return [(x, y) for (x, y) in lms]
    except:
        print("No face detected for img")
        return None

if __name__ == "__main__":

    


    app = FaceAnalysis(name="buffalo_l")   # best default model
    app.prepare(ctx_id=0, det_size=(112,112))
    get_landmarks_fn = insightface_landmarks_fn
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        RemoveEyes(get_landmarks_fn),
        #transforms.Resize((112, 112)),
    ])
    
    dataset = ScratchDataset(root="/DATA/faces/lfw/lfw-deepfunneled/lfw-deepfunneled/Mark_Philippoussis", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch_num = 0
    for batch in enumerate(dataloader):
        if batch is None:
            print("batch num", batch_num, "is None")
            continue
        for j,img in enumerate(batch):

            cv2.imwrite(f"output_{batch_num}_{j}.jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        batch_num += 1