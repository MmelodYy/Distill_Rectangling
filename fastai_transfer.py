from fastai.vision import *

PATH="data"
data = ImageDataBunch.from_folder(PATH,num_workers=4,ds_tfms = get_transforms(do_flip=True),size = 24)
learner = create_cnn(data,models.resnet18,pretrained=True,metrics = accuracy)
learner.freeze_to(-9)
learner.fit(10)