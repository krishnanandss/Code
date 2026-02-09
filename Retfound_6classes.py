import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             recall_score, f1_score, cohen_kappa_score)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")

# ============================================
# 0. CONFIGURATION & LOCAL SETUP
# ============================================
BASE_DIR = '/home/ncrai/krishnanand/6classes'

# Check for NVIDIA GPU
if not torch.cuda.is_available():
    print("WARNING: NVIDIA GPU not detected. Training will be slow on CPU.")
    DEVICE = torch.device("cpu")
else:
    print(f"SUCCESS: NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device("cuda")

sys.path.append(BASE_DIR)
try:
    import models_vit
except ImportError:
    sys.exit("File or model not found")

# --- HYPERPARAMETERS ---
ENABLE_OPTUNA = False
OPTUNA_TRIALS = 10
DEFAULT_LR = 0.001  # Higher initial LR when using cosine decay
DEFAULT_LR_MIN = 0.00001  # Minimum LR for cosine annealing
DEFAULT_BATCH_SIZE = 16
RETFOUND_WEIGHTS_PATH = os.path.join(BASE_DIR, 'RETFound_cfp_weights.pth')

# ============================================
# 1. PATHS
# ============================================
TRAIN_IMG_ROOT = os.path.join(BASE_DIR, 'Dataset/train')
TEST_IMG_ROOT  = os.path.join(BASE_DIR, 'Dataset/test')
CSV_PATH = os.path.join(BASE_DIR, 'Dataset/train.csv')
TEST_CSV_PATH = os.path.join(BASE_DIR, 'Dataset/test.csv')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'Model_Saved')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_retfound_frozen.pth')

if not os.path.exists(TRAIN_IMG_ROOT):
    sys.exit(f"Error: Training image folder not found at {TRAIN_IMG_ROOT}")
if not os.path.exists(TEST_IMG_ROOT):
    sys.exit(f"Error: Test image folder not found at {TEST_IMG_ROOT}")

# ============================================
# 2. UTILITY FUNCTIONS
# ============================================
def clean_label(val):
    val = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', "").strip()
    return val.split(',')[0].strip() if ',' in val else val

def load_preprocessed_image(image_path):
    try:
        return Image.open(image_path).convert('RGB')
    except:
        return Image.new('RGB', (224, 224), (0, 0, 0))

# ============================================
# 3. CORRECTED ARCHITECTURE (FROZEN RETFOUND)
# ============================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight, self.gamma, self.alpha, self.reduction = weight, gamma, alpha, reduction
    
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None: 
            focal_loss = self.alpha * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class FrozenRETFound(nn.Module):
    """
    Simplified RETFound model with permanently frozen backbone.
    
    FIX 1: Optimizer only includes trainable parameters (classifier)
    FIX 2: ViT backbone always stays in eval() mode to prevent BatchNorm/Dropout issues
    """
    def __init__(self, num_classes, checkpoint_path=None):
        super(FrozenRETFound, self).__init__()
        
        # Load RETFound ViT backbone
        self.vit = models_vit.vit_large_patch16(
            num_classes=num_classes, 
            drop_path_rate=0.1, 
            global_pool=True
        )
        
        # Remove the original classification head
        self.vit.head = nn.Identity()
        
        # Load pretrained weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading RETFound weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint_model = checkpoint['model']
            
            # Remove head weights from checkpoint to avoid dimension mismatch
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model: 
                    del checkpoint_model[k]
            
            self.vit.load_state_dict(checkpoint_model, strict=False)
            print("✓ RETFound weights loaded successfully")
        else:
            print("WARNING: RETFound weights not found. Initializing with random weights.")
        
        # FIX 1: FREEZE the entire ViT backbone permanently
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # FIX 2: Set ViT to eval mode permanently to handle BatchNorm/Dropout correctly
        self.vit.eval()
        
        print("✓ RETFound backbone FROZEN (requires_grad=False, eval mode)")
        
        # Simplified classifier (only trainable part)
        vit_feature_dim = 1024  # RETFound ViT-Large output dimension
        self.classifier = nn.Sequential(
            nn.Linear(vit_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"✓ Trainable classifier initialized: {vit_feature_dim} → 512 → 256 → {num_classes}")

    def train(self, mode=True):
        """
        Override train() to keep ViT in eval mode while classifier trains.
        
        FIX 2: This prevents BatchNorm from using batch statistics and 
        Dropout from randomly dropping features in the frozen backbone.
        """
        super(FrozenRETFound, self).train(mode)
        # Always keep ViT in eval mode
        self.vit.eval()
        return self

    def forward(self, images):
        """
        Forward pass using only image features from frozen RETFound.
        """
        # Extract features from frozen ViT backbone
        # No need for torch.no_grad() since ViT is in eval mode and frozen
        img_feat = self.vit.forward_features(images)
        
        # Extract CLS token if output is 3D sequence
        if img_feat.dim() == 3:
            img_feat = img_feat[:, 0, :]
        
        # Pass through trainable classifier
        return self.classifier(img_feat)


# ============================================
# 4. SIMPLIFIED DATASET (IMAGE-ONLY)
# ============================================
class SimpleOcularDataset(Dataset):
    """
    Dataset that only loads images (no handcrafted features).
    """
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_preprocessed_image(os.path.join(self.img_dir, row['filename']))
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['label_encoded'], dtype=torch.long)
        
        return image, label


# ============================================
# 5. DATA LOADING
# ============================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

if not os.path.exists(CSV_PATH): 
    sys.exit(f"Error: CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df['labels'] = df['labels'].apply(clean_label)

known_classes = ['N', 'D', 'G', 'C', 'A', 'M'] 
df = df[df['labels'].isin(known_classes)]

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['labels'])

print(f"Total samples: {len(df)}")
print(f"Classes: {le.classes_}")
print(f"Class distribution:\n{df['labels'].value_counts()}")

# Split into train/validation
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['label_encoded'], 
    random_state=42
)

print(f"\nTrain samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Compute class weights for handling imbalance
class_weights = torch.tensor(
    compute_class_weight(
        'balanced', 
        classes=np.unique(train_df['label_encoded']), 
        y=train_df['label_encoded']
    ), 
    dtype=torch.float
).to(DEVICE)

print(f"Class weights: {class_weights}")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced to 0.1 for medical images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ============================================
# 6. OPTUNA OPTIMIZATION
# ============================================
BEST_LR = DEFAULT_LR
BEST_BS = DEFAULT_BATCH_SIZE

def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    train_loader = DataLoader(
        SimpleOcularDataset(train_df, TRAIN_IMG_ROOT, train_transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        SimpleOcularDataset(val_df, TRAIN_IMG_ROOT, val_transform),
        batch_size=batch_size,
        num_workers=2
    )
    
    model = FrozenRETFound(
        num_classes=len(known_classes),
        checkpoint_path=RETFOUND_WEIGHTS_PATH
    ).to(DEVICE)
    
    # FIX 1: Only optimize trainable parameters (classifier)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    criterion = FocalLoss(weight=class_weights)
    
    # Quick training for 2 epochs on limited batches
    model.train()  # Classifier in train mode, ViT stays in eval mode
    limit_batches = 30
    
    for epoch in range(2):
        for i, (imgs, lbs) in enumerate(train_loader):
            if i >= limit_batches: 
                break
            
            imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
            
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, lbs)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (imgs, lbs) in enumerate(val_loader):
            if i >= limit_batches: 
                break
            
            imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            total += lbs.size(0)
            correct += (preds == lbs).sum().item()
    
    return correct / (total + 1e-6)


if ENABLE_OPTUNA:
    print("\n" + "="*60)
    print(f"RUNNING OPTUNA HYPERPARAMETER OPTIMIZATION ({OPTUNA_TRIALS} TRIALS)")
    print("="*60)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    
    print(f"\nBest hyperparameters: {study.best_params}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    
    BEST_LR = study.best_params['lr']
    BEST_BS = study.best_params['batch_size']
    USE_SCHEDULER = False  # Optuna handles LR, no scheduler needed
else:
    print("\nOptuna disabled. Using default hyperparameters with cosine annealing.")
    print(f"Initial LR: {DEFAULT_LR}, Min LR: {DEFAULT_LR_MIN}, Batch Size: {DEFAULT_BATCH_SIZE}")
    USE_SCHEDULER = True  # Enable cosine annealing scheduler


# ============================================
# 7. FINAL TRAINING
# ============================================
print("\n" + "="*60)
print(f"STARTING FINAL TRAINING")
print(f"Learning Rate: {BEST_LR}")
print(f"Batch Size: {BEST_BS}")
print(f"Epochs: 120 (with early stopping)")
print(f"RETFound: PERMANENTLY FROZEN + EVAL MODE")
print("="*60)

train_loader = DataLoader(
    SimpleOcularDataset(train_df, TRAIN_IMG_ROOT, train_transform),
    batch_size=BEST_BS,
    shuffle=True,
    drop_last=True,
    num_workers=4
)

val_loader = DataLoader(
    SimpleOcularDataset(val_df, TRAIN_IMG_ROOT, val_transform),
    batch_size=BEST_BS,
    num_workers=4
)

model = FrozenRETFound(
    num_classes=len(known_classes),
    checkpoint_path=RETFOUND_WEIGHTS_PATH
).to(DEVICE)

# Training configuration
EPOCHS = 200
best_val_acc = 0.0
patience_counter = 0
patience_limit = 22

# FIX 1: Only trainable parameters (classifier) in optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=BEST_LR
)
criterion = FocalLoss(weight=class_weights)

# Cosine Annealing LR Scheduler (only when Optuna is disabled)
if USE_SCHEDULER:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS, 
        eta_min=DEFAULT_LR_MIN
    )
    print(f"✓ Cosine Annealing LR Scheduler enabled: {BEST_LR} → {DEFAULT_LR_MIN}")
else:
    scheduler = None
    print(f"✓ Using fixed LR from Optuna: {BEST_LR}")

train_losses, val_losses, train_accs, val_accs = [], [], [], []

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
frozen_params = total_params - trainable_params

print(f"\nParameter Summary:")
print(f"  Total parameters:      {total_params:,}")
print(f"  Frozen parameters:     {frozen_params:,} (RETFound backbone)")
print(f"  Trainable parameters:  {trainable_params:,} (Classifier only)")
print(f"  Trainable %:           {100*trainable_params/total_params:.2f}%")

for epoch in range(EPOCHS):
    # ============================================
    # TRAINING PHASE
    # ============================================
    model.train()  # FIX 2: ViT stays in eval() due to overridden train() method
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for imgs, lbs in loop:
        imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += lbs.size(0)
        correct += (preds == lbs).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=correct/total)
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # ============================================
    # VALIDATION PHASE
    # ============================================
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for imgs, lbs in val_loader:
            imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
            outputs = model(imgs)
            val_loss += criterion(outputs, lbs).item()
            
            _, preds = torch.max(outputs, 1)
            val_total += lbs.size(0)
            val_correct += (preds == lbs).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{EPOCHS}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
          f"LR: {current_lr:.6f}")
    
    # Step the scheduler if enabled
    if scheduler is not None:
        scheduler.step()
    
    # ============================================
    # MODEL CHECKPOINTING
    # ============================================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'class_names': le.classes_,
            'num_classes': len(known_classes)
        }, MODEL_SAVE_PATH)
        
        print(f"✓ Saved Best Model (Val Acc: {best_val_acc:.4f})")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience_limit})")
        
        if patience_counter >= patience_limit:
            print(f"\n⚠ Early Stopping Triggered at Epoch {epoch+1}")
            break

print(f"\n{'='*60}")
print(f"TRAINING COMPLETED")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"{'='*60}")


# ============================================
# 8. TRAINING VISUALIZATION
# ============================================
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy', color='blue', linewidth=2)
plt.plot(val_accs, label='Val Accuracy', color='orange', linewidth=2)
plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', color='red', linewidth=2)
plt.plot(val_losses, label='Val Loss', color='green', linewidth=2)
plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# 9. EXTERNAL TEST SET EVALUATION
# ============================================
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

if not os.path.exists(TEST_CSV_PATH):
    print(f"Warning: Test CSV not found at {TEST_CSV_PATH}")
    print("Skipping test evaluation.")
else:
    df_test = pd.read_csv(TEST_CSV_PATH)
    df_test['labels'] = df_test['labels'].apply(clean_label)
    df_test = df_test[df_test['labels'].isin(known_classes)]
    df_test['label_encoded'] = le.transform(df_test['labels'])
    
    print(f"Test samples: {len(df_test)}")
    
    if os.path.exists(MODEL_SAVE_PATH):
        # Load best model
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=False)
        
        model = FrozenRETFound(num_classes=len(known_classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        print("✓ Best model loaded for testing")
        
        test_loader = DataLoader(
            SimpleOcularDataset(df_test, TEST_IMG_ROOT, transform=val_transform),
            batch_size=16,
            shuffle=False,
            num_workers=4
        )
        
        test_preds = []
        test_labels = []
        
        print("\nRunning predictions with Test-Time Augmentation (Horizontal Flip)...")
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # Standard prediction
                out1 = model(images)
                
                # TTA: Horizontal flip
                out2 = model(TF.hflip(images))
                
                # Average probabilities
                avg_prob = (F.softmax(out1, dim=1) + F.softmax(out2, dim=1)) / 2.0
                _, preds = torch.max(avg_prob, 1)
                
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        # ============================================
        # METRICS
        # ============================================
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            test_labels, 
            test_preds, 
            target_names=le.classes_,
            digits=4
        ))
        
        # Overall metrics
        test_acc = accuracy_score(test_labels, test_preds)
        test_recall = recall_score(test_labels, test_preds, average='weighted')
        test_f1 = f1_score(test_labels, test_preds, average='weighted')
        test_kappa = cohen_kappa_score(test_labels, test_preds)
        
        print("\n" + "="*60)
        print("OVERALL TEST METRICS")
        print("="*60)
        print(f"Accuracy:       {test_acc:.4f}")
        print(f"Recall:         {test_recall:.4f}")
        print(f"F1-Score:       {test_f1:.4f}")
        print(f"Cohen's Kappa:  {test_kappa:.4f}")
        
        # ============================================
        # CONFUSION MATRIX
        # ============================================
        cm = confusion_matrix(test_labels, test_preds)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix: Frozen RETFound Model', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print(f"Error: Saved model not found at {MODEL_SAVE_PATH}")

print("\n" + "="*60)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*60)