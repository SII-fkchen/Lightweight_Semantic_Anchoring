# finetune.py
# Fine-tuning script: loads the pretrained dual-encoder and fine-tunes it on a
# mixture of real and sampled synthetic word images, evaluating in-domain and
# out-of-domain retrieval after every epoch.

import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import DistilBertTokenizer

from model import DualEncoder
from loss import InvarianceLoss
from dataset import VisualVocabularyDataset, UniqueClassBatchSampler

CONFIG = {
    # Path configuration (fill in before running)
    "pretrained_ckpt": "",
    "save_dir": "./checkpoints/finetune_mixed",
    "json_path": "",

    # Training data sources
    "real_train_root": "",
    "synth_train_root": "",

    # Test data sources
    "in_domain_test_root": "",
    "ood_test_root": "",

    # Training hyperparameters
    "epochs": 20,
    "batch_size": 128,
    "num_workers": 8,
    "embedding_dim": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Mixing strategy: number of synthetic images sampled per run
    "synth_sample_size": 20000,
}


@torch.no_grad()
def build_gallery(model, json_path, device):
    """Build the text retrieval gallery from the vocabulary."""
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    with open(json_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    embeds = []
    ids = []

    for item in vocab:
        text = item['eng'].lower().strip()
        wid = item['id']

        inputs = tokenizer(text, return_tensors="pt", padding='max_length', max_length=32, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)

        emb = model.text_encoder(input_ids, mask)
        embeds.append(emb.cpu())
        ids.append(wid)

    return torch.cat(embeds, dim=0).to(device), torch.tensor(ids).to(device)


@torch.no_grad()
def evaluate(model, data_root, gallery_embeds, gallery_ids, device, task_name):
    """Evaluate image-to-text retrieval (R@1 / R@5) on a test split."""
    model.eval()
    dataset = VisualVocabularyDataset(data_root, CONFIG['json_path'], mode='test')
    if len(dataset) == 0:
        return 0.0

    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'] * 2, shuffle=False, num_workers=CONFIG['num_workers'])

    img_embeds_list = []
    labels_list = []

    for batch in loader:
        imgs = batch['image'].to(device)
        lbls = batch['label']
        embs = model.vision_encoder(imgs)
        img_embeds_list.append(embs)
        labels_list.append(lbls)

    img_embeds = torch.cat(img_embeds_list, dim=0)
    query_labels = torch.cat(labels_list, dim=0).to(device)

    sim_matrix = torch.matmul(img_embeds, gallery_embeds.T)

    _, indices = torch.topk(sim_matrix, k=5, dim=1)
    pred_ids = gallery_ids[indices]

    targets = query_labels.view(-1, 1)

    correct_1 = (pred_ids[:, 0] == targets.squeeze()).float().sum().item()
    r1 = correct_1 / len(query_labels)

    hits_5 = (pred_ids == targets).any(dim=1).float().sum().item()
    r5 = hits_5 / len(query_labels)

    print(f"[{task_name}] R@1: {r1*100:.2f}% | R@5: {r5*100:.2f}%")
    return r1


def main():
    """Fine-tuning main loop with mixed real + synthetic data."""
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")

    print(f"Loading Pretrained: {CONFIG['pretrained_ckpt']}")
    model = DualEncoder(embedding_dim=128)
    checkpoint = torch.load(CONFIG['pretrained_ckpt'], map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)

    print("Preparing Datasets...")
    real_dataset = VisualVocabularyDataset(CONFIG['real_train_root'], CONFIG['json_path'], mode='train')

    synth_dataset = VisualVocabularyDataset(CONFIG['synth_train_root'], CONFIG['json_path'], mode='train')
    synth_indices = torch.randperm(len(synth_dataset))[:CONFIG['synth_sample_size']]
    synth_subset = Subset(synth_dataset, synth_indices)

    mixed_dataset = ConcatDataset([real_dataset, synth_subset])

    mixed_labels = [int(wid) for _, wid in real_dataset.samples] + \
                   [int(synth_dataset.samples[int(i)][1]) for i in synth_indices]
    batch_sampler = UniqueClassBatchSampler(mixed_labels, batch_size=CONFIG['batch_size'], drop_last=True)

    train_loader = DataLoader(mixed_dataset, batch_sampler=batch_sampler, num_workers=CONFIG['num_workers'])

    print(f"Training Set: {len(real_dataset)} Real + {len(synth_subset)} Synthetic = {len(mixed_dataset)} Total Images")

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = InvarianceLoss(lambda_inv=0.5).to(device)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12], gamma=0.5)

    gallery_embeds, gallery_ids = build_gallery(model, CONFIG['json_path'], device)

    best_ood_r1 = 0.0

    print("\n>>> Start Mixed Fine-tuning <<<")

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0

        current_lr = optimizer.param_groups[0]['lr']

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [LR={current_lr:.1e}]")

        for batch in pbar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                img_emb, text_emb, logit_scale = model(images,input_ids,mask)
                loss = criterion(img_emb,text_emb,labels,logit_scale)
                # img_emb, text_emb, _ = model(images, input_ids, mask)
                # loss = criterion(img_emb, text_emb, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()

        print(f"\nEvaluating Epoch {epoch}...")

        gallery_embeds, gallery_ids = build_gallery(model, CONFIG['json_path'], device)

        r1_in = evaluate(model, CONFIG['in_domain_test_root'], gallery_embeds, gallery_ids, device, "In-Domain")
        r1_ood = evaluate(model, CONFIG['ood_test_root'], gallery_embeds, gallery_ids, device, "Out-of-Domain")

        save_path = os.path.join(CONFIG['save_dir'], f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ood_r1': r1_ood,
            'in_r1': r1_in
        }, save_path)

        if r1_ood > best_ood_r1:
            best_ood_r1 = r1_ood
            best_path = os.path.join(CONFIG['save_dir'], "best_finetuned_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                }, best_path)
            print(f"New Best OOD Model Saved! (R@1: {best_ood_r1:.2%})")

    print("\nFine-tuning Complete!")


if __name__ == "__main__":
    main()
