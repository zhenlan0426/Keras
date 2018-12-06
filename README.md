# Keras
https://github.com/uploadcare/pillow-simd

https://www.depends-on-the-definition.com/test-time-augmentation-keras/

https://stackoverflow.com/questions/38972380/keras-how-to-use-fit-generator-with-multiple-outputs-of-different-type

https://github.com/keras-team/keras/issues/8130

https://jkjung-avt.github.io/keras-image-cropping/

- build model to predict up-side-down image
- look into face recognition 
- possible aug: shift up/down a little, rotate a little, perspective, contrast
- fine-tune examples that are different classes but similar in model's eyes
- asymmetric margin, more margain for different class and less for same class
- two heads, one for feature embedding, one for weights for feature (for some pic, certain part might be unclear hence should have less weight)

import pandas as pd
dict_ = pd.read_csv('../input/train.csv').groupby(['Id']).apply(lambda x:list(x.Image)).reset_index()
dict_.columns = ['Id', 'Imgs']
dict_['length'] = [len(imgs) for imgs in dict_.Imgs.tolist()]
