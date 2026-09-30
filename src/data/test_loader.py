from loaders import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders()

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape) 

print("Classes:", train_loader.dataset.classes)
print("Class mapping:", train_loader.dataset.class_to_idx)