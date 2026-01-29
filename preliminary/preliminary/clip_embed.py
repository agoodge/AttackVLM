#%%
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import clip
import matplotlib.pyplot as plt
import torch.nn.functional as F

def encode_image(images):
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features

# ============================
# PGD attack (untargeted)
# ============================

def pgd_attack(images, labels, epsilon):
    alpha = ALPHA_FACTOR * epsilon
    delta = torch.rand_like(images)*(2*epsilon) - epsilon
    delta.requires_grad = True

    step_acc = []
    mean = torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor(STD,  device=DEVICE).view(1, 3, 1, 1)

    for _ in range(PGD_STEPS):
        adv_images = images + delta
        image_features = encode_image(adv_images)
        logits = image_features @ text_features.T

        # Untargeted loss: reduce similarity to correct label
        #loss = -logits.gather(1, labels.view(-1, 1)).mean()
        loss = F.cross_entropy(logits, labels)

        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        with torch.no_grad():
            delta = delta + alpha * delta.grad.sign()
            # limit perturbation to epsilon-ball
            delta = torch.clamp(delta, -epsilon, epsilon)
            # projection step
            delta = torch.clamp(images + delta, (0 - mean) / std, (1 - mean) / std) - images
        delta.requires_grad_(True)

    return images + delta

# ============================
# Config
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(1)
MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 128
PGD_STEPS = 15

PIXEL_EPSILONS = 4   # pixel-space perturbations
ALPHA_FACTOR = 0.3     # alpha = fraction of epsilon

# CLIP normalization
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD  = (0.26862954, 0.26130258, 0.27577711)
epsilon = torch.tensor([4 / 255.0 / s for s in STD], device=DEVICE).view(1, 3, 1, 1)

# ============================
# Load CLIP
# ============================
model, _ = clip.load(MODEL_NAME, device=DEVICE)
model.eval()

# ============================
# Dataset (CIFAR-10)
# ============================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

testset = datasets.CIFAR10(
    root="preliminary/data", train=False, download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False
)

# ============================
# Zero-shot text classifier
# ============================
classes = testset.classes
prompts = [f"a photo of a {c}" for c in classes]

text_tokens = clip.tokenize(prompts).to(DEVICE)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

num_classes = len(classes)

clean_image_features = []
adv_image_features = []
img_labels = []

for batch_idx, (images, labels) in enumerate(tqdm(testloader)):

    img_labels.append(labels)

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # PGD attack
    images_adv = pgd_attack(images, labels, epsilon)

    with torch.no_grad():
        clean_image_features.append(encode_image(images))
        adv_image_features.append(encode_image(images_adv))

labels = torch.cat(img_labels, dim=0)
clean_image_features = torch.cat(clean_image_features, dim=0)
adv_image_features = torch.cat(adv_image_features, dim=0)

clean_logits = clean_image_features @ text_features.T
adv_logits = adv_image_features @ text_features.T

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- confusion matrix ---
labels_np = labels.cpu().numpy()
preds_np = adv_logits.argmax(dim=1).cpu().numpy()

conf_mat = confusion_matrix(
    labels_np,
    preds_np,
    labels=np.arange(10)
)

# --- normalize (row-wise) ---
conf_mat_norm = conf_mat / conf_mat.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 6))
sns.heatmap(
    conf_mat_norm,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    yticklabels=testset.classes,
    xticklabels=testset.classes,
    vmin=0,
    vmax=1,
    cbar=True
)

plt.ylabel("True label")
plt.title("Attacked (4pix)")
plt.tight_layout()

col_sum = conf_mat_norm.mean(axis=0, keepdims=True)  # (1, 10)

plt.figure(figsize=(6, 2))
sns.heatmap(
    col_sum,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    yticklabels=[],
    xticklabels=testset.classes,
    cbar=False
)
plt.xlabel("Predicted label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# %%
sim = adv_image_features @ text_features.T
sim_mean = sim.mean(axis=0)

#%%

idx = torch.argwhere(clean_logits.argmax(dim=1).cpu() == labels).squeeze()
top2_pred = torch.topk(clean_logits[idx], k=2).indices[:,1]
adv_pred = adv_logits[idx].argmax(dim=1)

#%%

import torch
import matplotlib.pyplot as plt

# clean_logits: [N, C]
# adv_logits:   [N, C]
# labels:       [N]

# 1. Select only cleanly-correct samples
idx = torch.argwhere(clean_logits.argmax(dim=1).cpu() == labels_np).squeeze()

# 2. Attacked top-1 predictions
adv_pred = adv_logits[idx].argmax(dim=1)  # [M]

# 3. Compute clean ranks (descending: rank 0 = highest logit)
clean_ranks = clean_logits[idx].argsort(dim=1, descending=True)  # [M, C]

# 4. Find rank of attacked prediction in clean logits
# rank_of_adv[i] = position of adv_pred[i] in clean_ranks[i]
rank_of_adv = (clean_ranks == adv_pred.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]

# 5. Count occurrences of each rank
num_classes = clean_logits.shape[1]
rank_counts = torch.bincount(rank_of_adv, minlength=num_classes)

# 6. Plot
plt.figure(figsize=(8, 4))
plt.bar(range(num_classes), rank_counts.cpu(), width = 0.9)
plt.xlabel("Clean image logit rank of adversarial image predicted label")
plt.ylabel("Number of samples")
plt.title("Where adversarial images predicted labels ranked in the logits of clean images")
plt.xticks(range(num_classes))
plt.tight_layout()
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example data
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=1000)  # Normally distributed data

# Create KDE plot
sns.kdeplot(clean_logits.mean(axis=1).cpu(), shade=True, color="blue", bw_adjust=1, label = 'Clean')  # bw_adjust controls smoothness
sns.kdeplot(adv_logits.mean(axis=1).cpu(), shade=True, color="red", bw_adjust=1, label = 'Attacked')
plt.title("Mean Logit")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

#
# %%
