import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # Keep for potential future use, though not directly used for saving
import seaborn as sns # Keep for potential future use
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix # confusion_matrix added
import gc
import time

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label_numeric'] # Ensure this column exists and is numeric
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, model_save_path='best_skin_model.pth'):
    device = get_device()
    model = model.to(device)
    
    since = time.time()
    
    best_model_wts = model.state_dict() # Initialize best model weights
    best_acc = 0.0 # Initialize best validation accuracy

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, model_save_path) # Save best model state_dict
                print(f"Best val Acc: {best_acc:.4f}. Saved model to {model_save_path}")
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # Configuration
    BASE_DATA_PATH = r"C:\skin_argumented"
    IMAGE_PART1_PATH = os.path.join(BASE_DATA_PATH, "HAM10000_images_part_1")
    IMAGE_PART2_PATH = os.path.join(BASE_DATA_PATH, "HAM10000_images_part_2")
    METADATA_PATH = os.path.join(BASE_DATA_PATH, "HAM10000_metadata.csv")
    
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 15 # As per your previous output
    LEARNING_RATE = 1e-4
    MODEL_SAVE_NAME = 'skin_resnet50_model_statedict.pth' # Define model save name
    
    # Data loading and preprocessing
    print("--- Starting Data Loading and Preprocessing ---")
    df_metadata = pd.read_csv(METADATA_PATH)
    
    def get_image_path(image_id):
        path1 = os.path.join(IMAGE_PART1_PATH, f"{image_id}.jpg")
        path2 = os.path.join(IMAGE_PART2_PATH, f"{image_id}.jpg")
        return path1 if os.path.exists(path1) else path2 if os.path.exists(path2) else None
    
    df_metadata['image_path'] = df_metadata['image_id'].apply(get_image_path)
    df_metadata.dropna(subset=['image_path'], inplace=True)
    
    disease_categories = ['mel', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
    df_metadata['label_text'] = df_metadata['dx'].apply(lambda x: 'diseased' if x in disease_categories else 'healthy')
    
    label_encoder = LabelEncoder()
    df_metadata['label_numeric'] = label_encoder.fit_transform(df_metadata['label_text'])
    
    # **ADDED: Print label encoder mapping**
    print(f"LabelEncoder mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    # Example output might be: LabelEncoder mapping: {'diseased': 0, 'healthy': 1}
    # Or: LabelEncoder mapping: {'healthy': 0, 'diseased': 1}
    # Note this down for app.py CLASS_NAMES configuration!

    # Splitting into train and test. In train_model, 'val' uses the test_loader.
    # For a robust setup, you'd have train, validation, and test sets.
    # Here, test_df is used as the validation set during training.
    train_df, test_df = train_test_split(df_metadata, test_size=0.2, random_state=42, stratify=df_metadata['label_numeric'])
    
    # Data transforms - ensure these match what app.py expects
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Added ColorJitter
            transforms.RandomRotation(20), # Added RandomRotation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([ # This will be used for the test_df as validation
            transforms.Resize((IMG_SIZE, IMG_SIZE)), # Changed from Resize(256), CenterCrop(224) for simplicity and direct match.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets and dataloaders
    train_dataset = SkinDataset(train_df, transform=data_transforms['train'])
    # Using test_df as validation set during training
    val_dataset = SkinDataset(test_df, transform=data_transforms['val']) 
    
    # Determine num_workers based on CPU count for safety
    num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0) # Avoid issues on single core or undetectable CPUs

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    # Using test_df as validation set during training
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) # Match your previous output
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: e.g., diseased and healthy
    
    # Loss and optimizer
    # Calculate class weights for imbalanced dataset
    # Ensure the order of class_counts matches label_encoder.classes_ for correct weight assignment
    # If label_encoder.classes_ is ['diseased', 'healthy'], then counts for 'diseased' then 'healthy'
    class_counts_series = df_metadata['label_text'].value_counts()
    # Reorder counts according to label_encoder.classes_ to ensure weights align with numeric labels 0, 1, ...
    ordered_class_counts = [class_counts_series[cls_name] for cls_name in label_encoder.classes_]

    # weights = 1.0 / torch.tensor(ordered_class_counts, dtype=torch.float) # Invert for more weight to minority
    # class_weights_tensor = weights / weights.sum() # Normalize
    
    # Alternative weighting: weight = total_samples / (num_classes * count_per_class)
    num_train_samples = len(train_dataset)
    num_classes = len(label_encoder.classes_)
    train_class_counts = [np.sum(train_df['label_text'] == cls_name) for cls_name in label_encoder.classes_]
    
    class_weights_list = [num_train_samples / (num_classes * count) if count > 0 else 0 for count in train_class_counts]
    class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float).to(get_device())
    print(f"Using class weights for CrossEntropyLoss: {class_weights_tensor}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    dataloaders = {'train': train_loader, 'val': val_loader} # Pass val_loader as 'val'
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_NAME)
    
    # model.state_dict() is already saved within train_model for the best model
    print(f"Best trained model state_dict saved to {MODEL_SAVE_NAME}")
    
    # Test the model (using the val_loader as the test set here for consistency with training loop)
    model.eval() # Set model to evaluation mode
    device_eval = get_device() # Get device again, just in case
    model = model.to(device_eval)
    
    all_preds = []
    all_labels = []
    
    print("\n--- Evaluating on Test/Validation Set ---")
    with torch.no_grad():
        for inputs, labels in val_loader: # Using val_loader which corresponds to test_df
            inputs = inputs.to(device_eval)
            labels = labels.to(device_eval)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report_target_names = list(label_encoder.classes_)
    print("\nTest Results (based on validation set used during training):")
    print(classification_report(all_labels, all_preds, target_names=report_target_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=report_target_names, yticklabels=report_target_names)
    plt.title('Confusion Matrix on Test/Validation Set')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig("pytorch_confusion_matrix.png") # Save the plot
    print("Confusion matrix saved as pytorch_confusion_matrix.png")
    # plt.show() # Comment out if running in a non-GUI environment

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()
