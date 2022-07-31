import torch
import pathlib
import config
from config import image_height, image_width
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import os
# from torchdata.datapipes.iter import IterableWrapper
from torchvision import datasets
import numpy as np

def load_and_preprocess_image(img_path):
    # read pictures
    img = default_loader(img_path)

    train_transform=transforms.Compose([
        transforms.Resize((config.image_width, config.image_height)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])
    img = train_transform(img)
    return img

def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/'))]
    
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    
    # dict: {label : index}
    #label_to_index = dict((label, index) for index, label in enumerate(label_names))
    label_to_index = {}
    for label, index in enumerate(label_names):
        if(index[:9] == 'City_road'):
            label_to_index.update({index[:9] : 0})
        elif(index[:3] == 'fog'):
            label_to_index.update({index[:3] : 1})
        elif(index[:4] == 'rain'):
            label_to_index.update({index[:4] : 2})
        else:
            label_to_index.update({index[:4] : 3})
    #print(label_to_index)
    #print(label_to_index)
    # get all images' labels
    #all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]
    all_image_label=[]
    for single_image_path in all_image_path:
        #print("the single image path is {}".format(single_image_path))
        # for root, dirs, files in os.walk(single_image_path):
        #     print("oh yes!")
        #     print("the root is {} , the dirs is {} and the files is {}".format(root, dirs, files))
        files = os.path.basename(single_image_path) # files 带后缀文件名
        files2 = files.split('.')[0]  # files2 不带后缀文件名

        if(files2[:9] == 'City_road'):
            all_image_label.append(label_to_index[files2[:9]])
        elif(files2[:3] == 'fog'):
            all_image_label.append(label_to_index[files2[:3]])
        elif(files2[:3] == 'rain'):
            all_image_label.append(label_to_index[files2[:4]])
        else:
            all_image_label.append(label_to_index[files2[:4]])
        

    #print(all_image_label)
    return all_image_path, torch.as_tensor(all_image_label)

def get_dataset(dataset_root_dir):
    all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # print(dataset_root_dir)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    # print("this is all_img_path {}".format(all_image_path))
    all_images = list(load_and_preprocess_image(each_path) for each_path in list(all_image_path))
    # load the dataset and preprocess images
    # image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
    # label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    # dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(all_images)
    # for each_path in list(all_image_path):
    #     image = load_and_preprocess_image(each_path)
    #     print("the type of image is {}".format(type(image)))
    
    # image_dataset = IterableWrapper(all_images)
    # label_dataset = IterableWrapper(all_image_label)
    image_dataset = all_images
    label_dataset = all_image_label
    #print("the length of label is {}.".format(len(label_dataset)))
    dataset = zip(image_dataset, label_dataset)


    return dataset, image_count

def generate_datasets():
    train_dataset, train_count = get_dataset(dataset_root_dir=config.train_dir)
    valid_dataset, valid_count = get_dataset(dataset_root_dir=config.valid_dir)
    test_dataset, test_count = get_dataset(dataset_root_dir=config.test_dir)
    # print("this is train_len {} and this is test_len {}".format(train_count, test_count))
        

    # read the original_dataset in the form of batch
    # train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    # valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    # test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)

    #return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count
    return train_dataset, train_count, test_dataset, test_count




if __name__ == '__main__':
    
    Custom_train, Custom_count1, Custom_test, Custom_count2 = generate_datasets()
    # print("the train label length is {} and test's is {}".format(Custom_count1, Custom_count2))
