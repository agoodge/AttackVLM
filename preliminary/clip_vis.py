#%%
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import clip
import matplotlib.pyplot as plt

# ============================
# CLIP prediction
# ============================
def clip_predict(images):
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features @ text_features.T

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
        logits = clip_predict(adv_images)

        # Untargeted loss: reduce similarity to correct label
        loss = -logits.gather(1, labels.view(-1, 1)).mean()

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

    return (images + delta).detach()

# ============================
# Config
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(1)
MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 128
PGD_STEPS = 15
MAX_BATCHES = 50

PIXEL_EPSILONS = [1,2,8]   # pixel-space perturbations
ALPHA_FACTOR = 0.3     # alpha = fraction of epsilon

# CLIP normalization
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD  = (0.26862954, 0.26130258, 0.27577711)

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

import clip
model, preprocess = clip.load("ViT-B/16", device=DEVICE)
model.eval()

def denormalize(img):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
    img = img.cpu() * std + mean
    return img.clamp(0,1)

# ----  Load CIFAR-10 test set ----
transform = transforms.Compose([transforms.Resize(224),   transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(mean=MEAN, std=STD)])  # keep images in [0,1] for plotting
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

img, label = testset[0]
img = img.unsqueeze(0).to(DEVICE)
img_plot = denormalize(img)[0].permute(1,2,0).numpy()

logits_clean = clip_predict(img)
pred_clean = logits_clean.argmax(dim=1).item()

plt.figure(figsize=(6,4))

plt.subplot(1,len(PIXEL_EPSILONS)+1,1)
plt.imshow(img_plot)
plt.title(f"Clean: {classes[pred_clean]} ({logits_clean[0,pred_clean]:.2f})", fontsize = 8)
plt.axis('off')

for i, eps_pix in enumerate(PIXEL_EPSILONS):
    epsilon = torch.tensor(
        [eps_pix / 255.0 / s for s in STD],
        device=DEVICE
    ).view(1, 3, 1, 1)

    img_attacked = pgd_attack(img, torch.tensor([label]).to(DEVICE), epsilon)

    logits_attacked = clip_predict(img_attacked)

    pred_attacked = logits_attacked.argmax(dim=1).item()
    
    img_att_plot = denormalize(img_attacked)[0].permute(1,2,0)
    img_att_plot = torch.clamp(img_att_plot, 0, 1).numpy()
    # ---- 6. Plot clean vs attacked side by side ----

    plt.subplot(1,len(PIXEL_EPSILONS)+1,i+2)
    plt.imshow(img_att_plot)
    plt.title(f"{eps_pix}px: {classes[pred_attacked]} ({logits_clean[0,pred_attacked]:.2f})", fontsize = 8)
    plt.axis('off')

plt.tight_layout()
plt.show()
# %%
