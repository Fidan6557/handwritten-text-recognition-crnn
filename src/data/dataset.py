import torch
import pandas as pd
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_handwritten_image


class IAMLineDataset(Dataset):
    """
    PyTorch Dataset for IAM line-level handwritten text recognition.
    """

    def __init__(self, csv_path, char_to_idx, image_width=512, image_height=64):
        self.data = pd.read_csv(csv_path)
        self.char_to_idx = char_to_idx
        self.image_width = image_width  
        self.image_height = image_height

    def __len__(self):
        return len(self.data)
    
    def encode_text(self, text):
        return [
            self.char_to_idx[char]
            for char in str(text)
            if char in self.char_to_idx
        ]
    
    def __getitem__(Self, idx):
        row = self.data.iloc[idx]

        image = preprocess_handwritten_image(
            row["image_path"],
            image_width=self.image_width,
            image_height=self.image_height,
        )

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        text = str(row["text"])
        label = torch.tensor(self.encode_text(text), dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "label_length": len(label),
            "text": text,
        }
    

def collate_fn(batch):
    """
    Custom collate function required for CTC loss.
    It stacks images but concatenates labels into one long tensor.
    """

    images = torch.stack([item["image"] for item in batch])
    labels = torch.cat([item["label"] for item in batch])
    label_lengths = torch.tensor(
        [item["label_length"] for item in batch],
        dtype=torch.long
    )
    texts = [item["text"] for item in batch]

    return {
        "images": images,
        "labels": labels,
        "label_lengths": label_lengths,
        "texts": texts,
    }