# dataset.py
# Dataset and sampling utilities: the multilingual visual-word dataset and a
# batch sampler that guarantees unique semantic IDs within every batch.

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer


class UniqueClassBatchSampler(Sampler):
    """Batch sampler that guarantees unique semantic IDs within each batch.

    Standard image-text contrastive (ITC) loss treats all off-diagonal pairs
    as negatives. When a batch contains multiple samples of the same word ID,
    same-class samples are wrongly pushed apart (false negatives). This
    sampler enforces that each ID appears at most once per batch.

    Logic:
        1. Group sample indices by semantic ID.
        2. Shuffle within each group every epoch.
        3. For each batch, draw batch_size distinct IDs from a random
           permutation and take one sample from each ID's group
           (reshuffling the group when exhausted).
    """

    def __init__(self, labels, batch_size, drop_last=True, seed=None):
        """
        Args:
            labels: semantic ID of every sample (aligned with dataset indices)
            batch_size: number of samples per batch (= distinct IDs per batch)
            drop_last: if True, epoch length is len(labels) // batch_size
            seed: optional random seed for reproducibility
        """
        self.labels = [int(l) for l in labels]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        self.id_to_indices = {}
        for idx, lab in enumerate(self.labels):
            self.id_to_indices.setdefault(lab, []).append(idx)
        self.unique_ids = list(self.id_to_indices.keys())

        if batch_size > len(self.unique_ids):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds the number of classes "
                f"({len(self.unique_ids)}); cannot guarantee unique IDs per batch."
            )

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        pools = {lab: idxs.copy() for lab, idxs in self.id_to_indices.items()}
        for idxs in pools.values():
            self.rng.shuffle(idxs)
        ptr = {lab: 0 for lab in pools}

        ids = self.unique_ids.copy()
        self.rng.shuffle(ids)
        pos = 0

        for _ in range(len(self)):
            if pos + self.batch_size > len(ids):
                self.rng.shuffle(ids)
                pos = 0
            batch_ids = ids[pos:pos + self.batch_size]
            pos += self.batch_size

            batch = []
            for lab in batch_ids:
                if ptr[lab] >= len(pools[lab]):
                    self.rng.shuffle(pools[lab])
                    ptr[lab] = 0
                batch.append(pools[lab][ptr[lab]])
                ptr[lab] += 1
            yield batch


class VisualVocabularyDataset(Dataset):
    """Multilingual visual-word dataset.

    Images are organized as <data_root>/<lang>/<word_id>/*.jpg, and a JSON
    vocabulary maps each word ID to its text in multiple languages. During
    training, one language is randomly picked per sample as the text anchor,
    teaching the model cross-lingual equivalence.
    """

    def __init__(self,
                 data_root,
                 json_path,
                 tokenizer_name='distilbert-base-multilingual-cased',
                 max_len=32,
                 mode='train'):
        """
        Args:
            data_root: image root directory (e.g. <...>/vocabulary_image/train)
            json_path: path to the vocabulary JSON file
            tokenizer_name: HuggingFace tokenizer name
            max_len: maximum text token length
            mode: 'train' (with random augmentation) or 'test'
        """
        self.data_root = data_root
        self.max_len = max_len
        self.mode = mode

        print(f"[{mode.upper()}] Loading vocabulary from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            vocab_list = json.load(f)

        self.id_to_text = {}
        for item in vocab_list:
            wid = str(item['id'])
            self.id_to_text[wid] = {
                'eng': item['eng'],
                'ch': item['ch'],
                'spa': item['spa']
            }

        self.samples = []
        print(f"[{mode.upper()}] Scanning images in {data_root}...")

        supported_langs = ['ch', 'eng', 'spa']
        for lang_folder in supported_langs:
            lang_path = os.path.join(data_root, lang_folder)
            if not os.path.exists(lang_path):
                continue

            for wid in os.listdir(lang_path):
                word_dir = os.path.join(lang_path, wid)
                if not os.path.isdir(word_dir):
                    continue

                for img_name in os.listdir(word_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append((os.path.join(word_dir, img_name), wid))

        print(f"[{mode.upper()}] Found {len(self.samples)} images.")

        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, word_id = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)

        target_lang = random.choice(['eng', 'ch', 'spa'])
        text_raw = self.id_to_text[word_id][target_lang]
        text_raw = text_raw.lower()

        encoding = self.tokenizer.encode_plus(
            text_raw,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(int(word_id), dtype=torch.long)
        }


def get_dataloader(data_root, json_path, batch_size=32, mode='train', num_workers=4, unique_id_batch=True):
    """DataLoader factory function.

    Args:
        unique_id_batch: in train mode, use UniqueClassBatchSampler so that
            every batch contains unique semantic IDs (no same-class false
            negatives in the ITC loss)
    """
    dataset = VisualVocabularyDataset(
        data_root=data_root,
        json_path=json_path,
        mode=mode
    )

    if mode == 'train' and unique_id_batch:
        labels = [int(wid) for _, wid in dataset.samples]
        batch_sampler = UniqueClassBatchSampler(labels, batch_size=batch_size, drop_last=True)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(mode == 'train')
        )

    return dataloader
