import torchvision.transforms as transforms

# default transform
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# data augmentation
default_augmentation = transforms.RandomChoice([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(90, 90)),
    transforms.RandomRotation(degrees=(-90, -90)),
    transforms.RandomRotation(degrees=(180, 180))
])


# === Adamatch용 Weak Augmentation ===
augmentation_weak = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(size=224, padding=4, padding_mode='reflect'),
    transforms.ToTensor(),
])

# === Adamatch용 Strong Augmentation ===
augmentation_strong = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(size=224, padding=4, padding_mode='reflect'),
    transforms.RandAugment(num_ops=2, magnitude=10),
    transforms.ToTensor(),
])