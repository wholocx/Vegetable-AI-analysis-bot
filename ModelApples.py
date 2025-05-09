import os
import matplotlib.pyplot as plt
from fastai.vision.all import *
from fastai.data.transforms import get_image_files


if __name__ == '__main__':    
    bs = 50  #batch size
    sz = 224 #image size
    ds_path = 'DataSet'

    classes = []
    # Classes of pictures initialisation
    for d in os.listdir(ds_path):
        if os.path.isdir(os.path.join(ds_path, d)) and not d.startswith('.'):
            classes.append(d) 
    counter = 0
    # Data Verification
    for c in classes:
        if c != "models":
            print ("Class:", c)
            joined_path = os.path.join(ds_path, c)
            file_name_list = os.listdir(joined_path)
            path_list = list(map(lambda x: os.path.join(joined_path, x), file_name_list)) 
            for file_name in verify_images(path_list):
                os.remove(file_name)
    
    # Data Visualization
    data  = ImageDataLoaders.from_folder(ds_path, item_tfms = Resize(244), batch_tfms=[*aug_transforms(size=(244,244)), Normalize.from_stats(*imagenet_stats)], bs = bs, valid_pct=0.2)
    print ("There are", len(data.train_ds), "training images and", len(data.valid_ds), "validation images.")
    # визуализация загруженной партии
    data.show_batch(max_n=16)
    plt.show()
    # Начало обучения через CNN (https://www.google.com/url?q=http%3A%2F%2Fcs231n.github.io%2Fconvolutional-networks%2F)
    learn = vision_learner(data, models.resnet34, pretrained=True, weights=ResNet34_Weights.DEFAULT, metrics=accuracy)
    learn.lr_find()
    plt.show()
    learn.fit_one_cycle(4, slice(1e-2,1e-1))
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.show()
    interp.most_confused(min_val=1)
    plt.show()
    learn.export()

    # path_test = "thesisProj\AnalysisPhoto"
    # files = get_image_files(path_test)
