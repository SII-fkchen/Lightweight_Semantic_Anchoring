# pretrain.py
# Pretraining script: trains the dual-encoder on the multilingual visual-word
# dataset with the invariance loss, and evaluates image-to-text retrieval
# (R@1 / R@5) against a fixed text gallery after every epoch.

import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

from model import DualEncoder
from loss import InvarianceLoss
from dataset import get_dataloader
from transformers import DistilBertTokenizer

CONFIG = {
    # Path configuration (fill in before running)
    "train_data_root": "",
    "test_data_root": "",
    "json_path": "",
    "save_dir": "./checkpoints",

    # Training hyperparameters
    "epochs": 20,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "lambda_inv": 0.5,
    "num_workers": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def save_checkpoint(model, optimizer, epoch, acc, save_path):
    """Save a training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': acc
    }, save_path)


@torch.no_grad()
def evaluate_retrieval(model, test_loader, vocab_loader, device):
    """Full image-to-text retrieval evaluation.

    Builds a text gallery from the whole vocabulary, encodes all test
    images as queries, and computes R@1 and R@5.
    """
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    with open(CONFIG['json_path'], 'r') as f:
        vocab = json.load(f)

    gallery_embeds = []
    gallery_labels = []

    for item in vocab:
        text = item['eng'].lower()
        inputs = tokenizer(text, return_tensors="pt", padding='max_length', max_length=32, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        emb = model.text_encoder(input_ids, attention_mask)
        gallery_embeds.append(emb)
        gallery_labels.append(item['id'])

    gallery_embeds = torch.cat(gallery_embeds, dim=0)
    gallery_labels = torch.tensor(gallery_labels).to(device)

    img_embeds = []
    img_labels = []

    for batch in tqdm(test_loader, desc="Eval"):
        imgs = batch['image'].to(device)
        lbls = batch['label'].to(device)

        embs = model.vision_encoder(imgs)
        img_embeds.append(embs)
        img_labels.append(lbls)

    img_embeds = torch.cat(img_embeds, dim=0)
    img_labels = torch.cat(img_labels, dim=0)

    sim_matrix = torch.matmul(img_embeds, gallery_embeds.T)

    _, topk_indices = torch.topk(sim_matrix, k=5, dim=1)
    pred_labels = gallery_labels[topk_indices]

    targets = img_labels.view(-1, 1)
    correct = (pred_labels == targets)

    r1 = correct[:, 0].float().mean().item()
    r5 = correct[:, :5].any(dim=1).float().mean().item()

    return r1, r5


def main():
    """Pretraining main loop."""
    if not os.path.exists(CONFIG['save_dir']):
        os.makedirs(CONFIG['save_dir'])

    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")

    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        CONFIG['train_data_root'],
        CONFIG['json_path'],
        batch_size=CONFIG['batch_size'],
        mode='train',
        num_workers=CONFIG['num_workers']
    )

    test_loader = get_dataloader(
        CONFIG['test_data_root'],
        CONFIG['json_path'],
        batch_size=CONFIG['batch_size'] * 2,
        mode='test',
        num_workers=CONFIG['num_workers']
    )

    print("Initializing Model...")
    model = DualEncoder(embedding_dim=128).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = InvarianceLoss(lambda_inv=CONFIG['lambda_inv']).to(device)
    scaler = torch.cuda.amp.GradScaler()

    best_r1 = 0.0

    print("\nStart Training...")
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for batch in pbar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                img_emb, text_emb, logit_scale = model(images,input_ids,attention_mask)
                loss = criterion(img_emb,text_emb,labels,logit_scale)
                # img_emb, text_emb, _ = model(images, input_ids, attention_mask)
                # loss = criterion(img_emb, text_emb, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        print(f"Validating Epoch {epoch}...")
        r1, r5 = evaluate_retrieval(model, test_loader, None, device)
        print(f"Epoch {epoch} Result: Loss={avg_loss:.4f}, R@1={r1:.4f}, R@5={r5:.4f}")

        if r1 > best_r1:
            best_r1 = r1
            save_path = os.path.join(CONFIG['save_dir'], "best_model.pth")
            save_checkpoint(model, optimizer, epoch, r1, save_path)
            print(f"New Best Model Saved! (R@1: {best_r1:.4f})")

    print(f"\nTraining Complete. Best R@1: {best_r1:.4f}")


if __name__ == "__main__":
    main()
