import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             recall_score, f1_score, cohen_kappa_score)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import h5py
import optuna  # Required for hyperparameter tuning
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")

# ============================================
# 0. CONFIGURATION & LOCAL SETUP
# ============================================
BASE_DIR = '/home/user1/krishnanand'

# Check for NVIDIA GPU
if not torch.cuda.is_available():
    print("WARNING: NVIDIA GPU not detected. Training will be slow on CPU.")
    DEVICE = torch.device("cpu")
else:
    print(f"SUCCESS: NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device("cuda")

try:
    import models_vit
except ImportError:
    sys.exit("File or model not found")

# --- HYPERPARAMETERS ---
ENABLE_OPTUNA = True  # Set to True to run hyperparameter tuning
OPTUNA_TRIALS = 10    # Number of trials to run
DEFAULT_LR = 0.0001
DEFAULT_BATCH_SIZE = 16
RETFOUND_WEIGHTS_PATH = os.path.join(BASE_DIR, 'RETFound_cfp_weights.pkl')

# ============================================
# 1. PATHS
# ============================================
TRAIN_IMG_ROOT = os.path.join(BASE_DIR, 'Dataset/Preprocessed_Images/train')
TEST_IMG_ROOT  = os.path.join(BASE_DIR, 'Dataset/Preprocessed_Images/test')
CSV_PATH = os.path.join(BASE_DIR, 'Dataset/train.csv')
TEST_CSV_PATH = os.path.join(BASE_DIR, 'Dataset/test.csv')
TRAIN_FEATURES_H5 = os.path.join(BASE_DIR, 'extracted/train_features.h5')
TEST_FEATURES_H5 = os.path.join(BASE_DIR, 'extracted/test_features.h5')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'Model_Saved')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_retfound_hybrid.pth')

if not os.path.exists(TRAIN_IMG_ROOT) or not os.path.exists(TEST_IMG_ROOT):
    sys.exit("Error: Image folders not found.")

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

def load_h5_features(h5_path):
    if not os.path.exists(h5_path):
        sys.exit(f"Error: H5 feature file not found at {h5_path}")
    with h5py.File(h5_path, 'r') as hf:
        feats = [hf[k][:] for k in ['vascular_features', 'lesion_features', 'anatomical_features',
                                    'color_features', 'glcm_features', 'lbp_features',
                                    'hog_features', 'gabor_features']]
        features = np.hstack(feats)
        filenames = hf['filenames'][:].astype(str)
        labels_encoded = hf['labels_encoded'][:]
        labels = hf['labels'][:].astype(str) if 'labels' in hf else None
    return features, filenames, labels_encoded, labels

# ============================================
# 3. ARCHITECTURE (RETFOUND)
# ============================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight, self.gamma, self.alpha, self.reduction = weight, gamma, alpha, reduction
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None: focal_loss = self.alpha * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class ViTHybrid(nn.Module):
    def __init__(self, num_classes, num_handcrafted_features, checkpoint_path=None):
        super(ViTHybrid, self).__init__()
        self.vit = models_vit.vit_large_patch16(num_classes=num_classes, drop_path_rate=0.1, global_pool=True)
        self.vit.head = nn.Identity() # Remove head to fix dimensions

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading RETFound weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_model = checkpoint['model']
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model: del checkpoint_model[k]
            self.vit.load_state_dict(checkpoint_model, strict=False)
        
        vit_feature_dim = 1024
        self.manual_mlp = nn.Sequential(
            nn.Linear(num_handcrafted_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(vit_feature_dim + 512, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, images, handcrafted_features):
        img_feat = self.vit.forward_features(images)
        if img_feat.dim() == 3: img_feat = img_feat[:, 0, :] # Extract CLS token
        man_feat = self.manual_mlp(handcrafted_features)
        return self.classifier(torch.cat([img_feat, man_feat], dim=1))

    def freeze_backbone(self):
        for param in self.vit.parameters(): param.requires_grad = False
    def unfreeze_all(self):
        for param in self.vit.parameters(): param.requires_grad = True

# ============================================
# 4. DATASET & LOADING
# ============================================
class HybridOcularDataset(Dataset):
    def __init__(self, dataframe, img_dir, handcrafted_features, feature_indices, transform=None):
        self.df, self.img_dir = dataframe.reset_index(drop=True), img_dir
        self.handcrafted_features, self.feature_indices, self.transform = handcrafted_features, feature_indices, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_preprocessed_image(os.path.join(self.img_dir, row['filename']))
        if self.transform: image = self.transform(image)
        f_idx = self.feature_indices.get(row['filename'], -1)
        features = self.handcrafted_features[f_idx] if f_idx != -1 else np.zeros(self.handcrafted_features.shape[1])
        return image, torch.tensor(features, dtype=torch.float32), torch.tensor(row['label_encoded'], dtype=torch.long)

print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

if not os.path.exists(CSV_PATH): sys.exit(f"Error: CSV not found at {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
df['labels'] = df['labels'].apply(clean_label)
known_classes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
df = df[df['labels'].isin(known_classes)]
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['labels'])

train_features, train_filenames, _, _ = load_h5_features(TRAIN_FEATURES_H5)
train_feature_indices = {fname: idx for idx, fname in enumerate(train_filenames)}
scaler = StandardScaler()
train_features_normalized = scaler.fit_transform(train_features)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)
class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_df['label_encoded']), y=train_df['label_encoded']), dtype=torch.float).to(DEVICE)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================
# 5. OPTUNA OPTIMIZATION
# ============================================
BEST_LR = DEFAULT_LR
BEST_BS = DEFAULT_BATCH_SIZE

def objective(trial):
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    
    # 2. Setup DataLoaders for this trial
    train_loader = DataLoader(HybridOcularDataset(train_df, TRAIN_IMG_ROOT, train_features_normalized, train_feature_indices, train_transform), 
                              batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(HybridOcularDataset(val_df, TRAIN_IMG_ROOT, train_features_normalized, train_feature_indices, val_transform), 
                            batch_size=batch_size, num_workers=2)
    
    # 3. Initialize Model
    model = ViTHybrid(len(known_classes), train_features_normalized.shape[1], checkpoint_path=RETFOUND_WEIGHTS_PATH).to(DEVICE)
    model.freeze_backbone()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(weight=class_weights)
    
    # 4. Short Training Loop (3 Epochs)
    model.train()
    limit_batches = 30 # Only run 30 batches per epoch to save time
    
    for epoch in range(2): # Run for 2 epochs
        for i, (imgs, feats, lbs) in enumerate(train_loader):
            if i >= limit_batches: break
            imgs, feats, lbs = imgs.to(DEVICE), feats.to(DEVICE), lbs.to(DEVICE)
            optimizer.zero_grad()
            outs = model(imgs, feats)
            loss = criterion(outs, lbs)
            loss.backward()
            optimizer.step()
            
    # 5. Validation Accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (imgs, feats, lbs) in enumerate(val_loader):
            if i >= limit_batches: break
            imgs, feats, lbs = imgs.to(DEVICE), feats.to(DEVICE), lbs.to(DEVICE)
            outs = model(imgs, feats)
            _, preds = torch.max(outs, 1)
            total += lbs.size(0)
            correct += (preds == lbs).sum().item()
            
    return correct / (total + 1e-6)

if ENABLE_OPTUNA:
    print("\n" + "="*60)
    print(f"RUNNING OPTUNA ({OPTUNA_TRIALS} TRIALS)")
    print("="*60)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    
    print("\nBest params found:")
    print(study.best_params)
    
    BEST_LR = study.best_params['lr']
    BEST_BS = study.best_params['batch_size']
else:
    print("Optuna disabled. Using default hyperparameters.")

# ============================================
# 6. FINAL TRAINING
# ============================================
print("\n" + "="*60)
print(f"STARTING FINAL TRAINING (LR={BEST_LR}, BS={BEST_BS})")
print("="*60)

train_loader = DataLoader(HybridOcularDataset(train_df, TRAIN_IMG_ROOT, train_features_normalized, train_feature_indices, train_transform), 
                          batch_size=BEST_BS, shuffle=True, drop_last=True, num_workers=4)
val_loader = DataLoader(HybridOcularDataset(val_df, TRAIN_IMG_ROOT, train_features_normalized, train_feature_indices, val_transform), 
                        batch_size=BEST_BS, num_workers=4)

model = ViTHybrid(len(known_classes), train_features_normalized.shape[1], checkpoint_path=RETFOUND_WEIGHTS_PATH).to(DEVICE)
model.freeze_backbone()

optimizer = optim.Adam(model.parameters(), lr=BEST_LR)
criterion = FocalLoss(weight=class_weights)

EPOCHS, UNFREEZE_EPOCH, best_val_acc, patience_counter = 120, 5, 0.0, 0
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(EPOCHS):
    if epoch == UNFREEZE_EPOCH:
        print(">>> Unfreezing RETFound backbone with 1e-6 LR")
        model.unfreeze_all()
        for pg in optimizer.param_groups: pg['lr'] = 1e-6

    model.train()
    r_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for imgs, feats, lbs in loop:
        imgs, feats, lbs = imgs.to(DEVICE), feats.to(DEVICE), lbs.to(DEVICE)
        optimizer.zero_grad()
        outs = model(imgs, feats)
        loss = criterion(outs, lbs)
        loss.backward()
        optimizer.step()
        
        r_loss += loss.item()
        _, preds = torch.max(outs, 1)
        total += lbs.size(0)
        correct += (preds == lbs).sum().item()
        loop.set_postfix(loss=loss.item(), acc=correct/total)

    train_losses.append(r_loss/len(train_loader))
    train_accs.append(correct/total)

    model.eval()
    v_loss, v_corr, v_tot = 0, 0, 0
    with torch.no_grad():
        for imgs, feats, lbs in val_loader:
            imgs, feats, lbs = imgs.to(DEVICE), feats.to(DEVICE), lbs.to(DEVICE)
            outs = model(imgs, feats)
            v_loss += criterion(outs, lbs).item()
            _, preds = torch.max(outs, 1)
            v_tot += lbs.size(0)
            v_corr += (preds == lbs).sum().item()

    val_acc = v_corr/v_tot
    val_losses.append(v_loss/len(val_loader))
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}: Train Acc {train_accs[-1]:.4f}, Val Acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler, 'class_names': le.classes_}, MODEL_SAVE_PATH)
        print(f"✓ Saved Best Model (Acc: {best_val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= 17:
            print("Early Stopping Triggered")
            break

# ============================================
# 7. VISUALIZATION
# ============================================
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy', color='blue')
plt.plot(val_accs, label='Val Accuracy', color='orange')
plt.title('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', color='red')
plt.plot(val_losses, label='Val Loss', color='green')
plt.title('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# ============================================
# 8. EXTERNAL TEST SET EVALUATION
# ============================================
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

if not os.path.exists(TEST_CSV_PATH): sys.exit(f"Error: Test CSV not found at {TEST_CSV_PATH}")
df_test = pd.read_csv(TEST_CSV_PATH)
df_test['labels'] = df_test['labels'].apply(clean_label)
df_test = df_test[df_test['labels'].isin(known_classes)]
df_test['label_encoded'] = le.transform(df_test['labels'])

test_features, test_filenames, _, _ = load_h5_features(TEST_FEATURES_H5)
df_test = df_test[df_test['filename'].isin(test_filenames)]

if os.path.exists(MODEL_SAVE_PATH):
    checkpoint = torch.load(MODEL_SAVE_PATH)
    scaler = checkpoint['scaler']
    test_features_normalized = scaler.transform(test_features)

    model = ViTHybrid(len(known_classes), test_features_normalized.shape[1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    test_loader = DataLoader(HybridOcularDataset(df_test, TEST_IMG_ROOT, test_features_normalized, test_feature_indices, transform=val_transform), 
                             batch_size=16, shuffle=False, num_workers=4)

    test_preds, test_labels = [], []
    print("Running Predictions with TTA (Horizontal Flip)...")
    with torch.no_grad():
        for images, features, labels in tqdm(test_loader):
            images, features, labels = images.to(DEVICE), features.to(DEVICE), labels.to(DEVICE)
            out1 = model(images, features)
            out2 = model(TF.hflip(images), features)
            avg_prob = (F.softmax(out1, dim=1) + F.softmax(out2, dim=1)) / 2.0
            _, preds = torch.max(avg_prob, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    print(classification_report(test_labels, test_preds, target_names=le.classes_))
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix: RETFound Hybrid Model'); plt.show()
else:
    print(f"Error: Saved model not found at {MODEL_SAVE_PATH}")

    