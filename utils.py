class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_dir:str, mode: str):
        self.df = dataframe
        self.mode = mode
        self.image_dir = image_dir
        
        transforms_list = []
        if self.mode == 'train':
            # Increase image size from (64,64) to higher resolution,
            # Make sure to change in RandomResizedCrop as well.
            transforms_list = [
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ]
        else:
            transforms_list.extend([
                # Keep this resize same as train
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index: int):
        image_id = self.df.iloc[index].id
        image_path = f"{self.image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
        image = Image.open(image_path)
        image = self.transforms(image)

        if self.mode == 'test' :
            return {'image':image}
        else:
            return {'image':image, 
                    'target':self.df.iloc[index].landmark_id}

    def __len__(self) -> int:
        return self.df.shape[0]
    
def load_data(train, val, test, train_dir, val_dir, test_dir):
    counts = train.landmark_id.value_counts()
    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    num_classes = selected_classes.shape[0]
    print('classes with at least N samples:', num_classes)

    train = train.loc[train.landmark_id.isin(selected_classes)]
    val = val.loc[val.landmark_id.isin(selected_classes)]
    print('train_df', train.shape)
    print('val_df', val.shape)
    print('test_df', test.shape)

    # filter non-existing test images
    exists = lambda img: os.path.exists(f'{test_dir}/{img[0]}/{img[1]}/{img[2]}/{img}.jpg')
    test = test.loc[test.id.apply(exists)]
    print('test_df after filtering', test.shape)

    label_encoder = LabelEncoder()
    label_encoder.fit(train.landmark_id.values)
    print('found classes', len(label_encoder.classes_))
    assert len(label_encoder.classes_) == num_classes

    train.landmark_id = label_encoder.transform(train.landmark_id)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(val.landmark_id.values)
    print('found classes', len(label_encoder.classes_))
    assert len(label_encoder.classes_) == num_classes

    val.landmark_id = label_encoder.transform(val.landmark_id)

    train_dataset = ImageDataset(train, train_dir, mode='train')
    val_dataset = ImageDataset(val, val_dir, mode='val')
    test_dataset = ImageDataset(test, test_dir, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, label_encoder, num_classes
