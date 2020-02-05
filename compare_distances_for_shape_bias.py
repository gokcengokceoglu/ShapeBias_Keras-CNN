from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
from keras.models import Model as keras_model
from keras import backend as K
import glob, os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
from keras.preprocessing import image
from itertools import combinations
import matplotlib.pyplot as plt


def get_trained_model(path_to_pretrained_model,layer_name='dense_2'):
        pre_trained_model = load_model(path_to_pretrained_model)
        model_top_removed = keras_model(inputs=pre_trained_model.input, outputs=pre_trained_model.get_layer(layer_name).output)
        return model_top_removed


model_paths = glob.glob(os.path.join('', '*.hdf5'))
test_data_path = 'test_data_colors_textures'
shape_classes = os.listdir(test_data_path)
test_set_imgs = []
predictions_arr = []
calculate_within_cls_similarity = 0
calculate_bw_cls_similarity = 1
within_cls_combinations = combinations([0,1,2,3],2)


between_cls_combinations = combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2)

test_data_list = []
# Each row corresponds to a shape class
# In each column, there are frames with a similar feature : texture or color
for shape_class in shape_classes :
    if shape_class.startswith('dataset'):
        imgs = glob.glob(os.path.join(test_data_path, shape_class, '*.png'))
        test_data_list.append(imgs)

if calculate_bw_cls_similarity :
    mean_epoch_cls_similarities =[]
    for model in model_paths:
        pretrained_model = get_trained_model(model,'dense_2')
        epoch_cls_similarity = []
        between_cls_combinations = combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2)
        for ii in list(between_cls_combinations) :
            print(ii)
            curr_test_cls1 = test_data_list[ii[0]]
            print(curr_test_cls1)
            curr_test_cls2 = test_data_list[ii[1]]
            print(curr_test_cls2)
            between_cls_similarities = []
            for test_img1_path, test_img2_path in zip(curr_test_cls1, curr_test_cls2) :
                img1 = (1. / 255) * np.array(Image.open(test_img1_path))
                img1 = image.img_to_array(img1)
                img1 = np.expand_dims(img1, axis=0)
                img2 = (1. / 255) * np.array(Image.open(test_img2_path))
                img2 = image.img_to_array(img2)
                img2 = np.expand_dims(img2, axis=0)
                feature_vect1 = pretrained_model.predict(img1)
                feature_vect2 = pretrained_model.predict(img2)
                similarity = cosine_similarity(feature_vect1, feature_vect2)
                similarity = similarity[0][0]
                between_cls_similarities.append(similarity)
            mean_bw_cls_similarity = np.mean(between_cls_similarities)
            print(mean_bw_cls_similarity)
            epoch_cls_similarity.append(mean_bw_cls_similarity)
            print(epoch_cls_similarity)
        print(model)
        mean_epoch_cls_similarity = np.mean(epoch_cls_similarity)
        print(mean_epoch_cls_similarity)
        mean_epoch_cls_similarities.append(mean_epoch_cls_similarity)
    plt.plot(mean_epoch_cls_similarities)
    plt.ylabel('Mean between class similarities')
    plt.xlabel('Epoch')
    plt.show()



if calculate_within_cls_similarity  :
    mean_epoch_cls_similarities = []
    for model in model_paths:
        pretrained_model = get_trained_model(model,'dense_2')
        epoch_cls_similarity =[]
        for shape_class in shape_classes :
            shape_cls_cosine_similarities = []
            if shape_class.startswith('dataset'):
                imgs_same_shape = glob.glob(os.path.join(test_data_path, shape_class, '*.png'))
                within_cls_combinations = combinations([0, 1, 2, 3], 2)
                for combination in within_cls_combinations :
                    print(combination)
                    img1_name = imgs_same_shape[combination[0]]
                    img2_name = imgs_same_shape[combination[1]]
                    img1 = (1. / 255) * np.array(Image.open(img1_name))
                    img1 = image.img_to_array(img1)
                    img1 = np.expand_dims(img1, axis=0)
                    img2 = (1. / 255) * np.array(Image.open(img2_name))
                    img2 = image.img_to_array(img2)
                    img2 = np.expand_dims(img2, axis=0)
                    feature_vect1 = pretrained_model.predict(img1)
                    feature_vect2 = pretrained_model.predict(img2)
                    similarity = cosine_similarity(feature_vect1,feature_vect2)
                    similarity = similarity[0][0]
                    shape_cls_cosine_similarities.append(similarity)
                    print('Similarity = '+ str(similarity))
                print('shape_class = ' + shape_class)
                mean_cls_similarity = np.mean(shape_cls_cosine_similarities)
                epoch_cls_similarity.append(mean_cls_similarity)
                print('mean_cls_similarity = ' + str(mean_cls_similarity))
        print('model = ' + model)
        print('epoch_cls_similarity = '+str(epoch_cls_similarity))
        mean_epoch_cls_similarity = np.mean(epoch_cls_similarity)
        print(mean_epoch_cls_similarity)
        mean_epoch_cls_similarities.append(mean_epoch_cls_similarity)
    plt.plot(mean_epoch_cls_similarities)
    plt.ylabel('Mean within class similarities')
    plt.xlabel('Epoch')
    plt.show()












