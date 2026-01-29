#%%
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import clip
import matplotlib.pyplot as plt
import torch.nn.functional as F


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

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            step_acc.append((preds == labels).float().mean().item())

    return images + delta, step_acc

# ============================
# Evaluation
# ============================
def evaluate(epsilon, num_classes):
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    step_acc_all = []

    for batch_idx, (images, labels) in enumerate(tqdm(testloader)):
        if batch_idx >= MAX_BATCHES:
            break

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        adv_images, step_acc = pgd_attack(images, labels, epsilon)
        step_acc_all.append(step_acc)

        # Final-step predictions only
        with torch.no_grad():
            preds = clip_predict(adv_images).argmax(dim=1)

        for c in range(num_classes):
            mask = (labels == c)
            total_per_class[c] += mask.sum().item()
            correct_per_class[c] += (preds[mask] == c).sum().item()

    class_acc = correct_per_class / total_per_class.clamp(min=1)
    avg_step_acc = torch.tensor(step_acc_all).mean(dim=0)

    return class_acc.numpy(), avg_step_acc.numpy()

# ============================
# Config
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(1)
MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 128
PGD_STEPS = 15
MAX_BATCHES = 50

PIXEL_EPSILONS = [1,2,4,8,16]   # pixel-space perturbations
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

correct_per_class = torch.zeros(num_classes)
total_per_class = torch.zeros(num_classes)

for batch_idx, (images, labels) in enumerate(tqdm(testloader)):
    if batch_idx >= MAX_BATCHES:
        break

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Final-step predictions only
    with torch.no_grad():
        preds = clip_predict(images).argmax(dim=1)

    for c in range(num_classes):
        mask = (labels == c)
        total_per_class[c] += mask.sum().item()
        correct_per_class[c] += (preds[mask] == c).sum().item()

class_acc = correct_per_class / total_per_class.clamp(min=1)
avg_acc = torch.tensor(class_acc).mean(dim=0)
print(f"Final adversarial accuracy: {avg_acc:.4f}")
print("Class-wise accuracy:")
for cls, acc in zip(classes, class_acc):
    print(f"  {cls:>10}: {acc:.3f}")

plt.figure(figsize=(7, 5))
steps = list(range(PGD_STEPS))

for eps_pix in PIXEL_EPSILONS:
    epsilon = torch.tensor(
        [eps_pix / 255.0 / s for s in STD],
        device=DEVICE
    ).view(1, 3, 1, 1)

    print(f"\nRunning PGD with ε = {eps_pix} pixels")
    class_acc, step_acc = evaluate(epsilon, num_classes)

    print(f"Final adversarial accuracy: {step_acc[-1]:.4f}")
    print("Class-wise accuracy:")
    for cls, acc in zip(classes, class_acc):
        print(f"  {cls:>10}: {acc:.3f}")

    plt.plot(steps, step_acc, marker="o", label=f"ε={eps_pix}px")

# ============================
# Plot
# ============================
plt.xlabel("PGD steps")
plt.ylabel("Accuracy")
plt.title("CLIP ViT-B/16 — PGD Attack on CIFAR-10")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
