from loaders import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders()

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)