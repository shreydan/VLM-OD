import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import json
import re
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from transformers import TrainerCallback, TrainerState, TrainerControl
import random

class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, augments=None, synonyms=None, use_synonyms=False):
        super().__init__(root, annFile)
        self.augments = augments
        self.num_tokens = 256
        self.synonyms = synonyms
        self.use_synonyms = use_synonyms

    def bbox_to_tokens(self,bbox,img_width,img_height):
        """
        (x,y,w,h) -> token[y_min, x_min, y_max, x_max]
        """
        x,y,w,h = bbox
        x_min = int(round((x / img_width) * (self.num_tokens-1)))
        x_max = int(round(((x+w) / img_width) * (self.num_tokens-1)))
        y_min = int(round((y / img_height) * (self.num_tokens-1)))
        y_max = int(round(((y+h) / img_height) * (self.num_tokens-1)))

        bbox = [y_min,x_min,y_max,x_max]

        assert not any(v>self.num_tokens-1 or v<0 for v in bbox), f'incorrect calculation: {bbox}'

        return [f"<loc{v:03d}>" for v in bbox]

    def tokens_to_bbox(self,tokens,img_width,img_height):
        """
        token[y_min, x_min, y_max, x_max] -> (x,y,w,h)
        """
        y_min, x_min, y_max, x_max = [int(re.search(r"\d+", t).group()) for t in tokens]

        assert not any(v>self.num_tokens-1 or v<0 for v in [y_min, x_min, y_max, x_max]), f'incorrect tokens: {tokens}'

        x_min = (x_min / (self.num_tokens-1)) * img_width
        y_min = (y_min / (self.num_tokens-1)) * img_height
        x_max = (x_max / (self.num_tokens-1)) * img_width
        y_max = (y_max / (self.num_tokens-1)) * img_height

        bbox = [round(x_min,2), round(y_min,2), round(x_max-x_min, 2), round(y_max-y_min,2)]
        return bbox


    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        image_np = np.array(image)

        bboxes = [obj['bbox'] for obj in target]
        category_ids = [obj['category_id'] for obj in target]

        augmented = self.augments(
            image=image_np,
            bboxes=bboxes,
            category_ids=category_ids
        )
        image = augmented['image']
        bboxes = augmented['bboxes']
        category_ids = augmented['category_ids']

        chosen_synonyms = {k:random.choice(v) if self.use_synonyms else v[0] for k,v in self.synonyms.items()}

        targets = dict()
        img_width, img_height = 512,512

        for i, bbox in enumerate(bboxes):
            category_label = chosen_synonyms[category_ids[i]]

            if targets.get(category_label,None) is None:
                targets[category_label] = list()

            tokens = self.bbox_to_tokens(bbox, img_width, img_height)
            text = f"{category_label} {''.join(tokens)};"
            targets[category_label].append(text)

        
        texts = []
        for label, label_texts in targets.items():
            random.shuffle(label_texts)
            full_text = '\n'.join(label_texts).strip()
            texts.append(full_text)

        text = '\n'.join(texts).strip() if len(texts) > 0 else 'no objects detected'

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"detect {'; '.join(targets.keys()).strip()}".strip()}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text}
                ]
            },
        ]

        return {
            'messages': conversation,
            'image': image
        }


base_path = Path('/data2/shreyas/datasets/coco/coco2017')
ds_info = {
    'train': {
        'dir': base_path / 'train2017',
        'annotation': base_path / 'annotations' / 'instances_train2017.json'
    },
    'val': {
        'dir': base_path / 'val2017',
        'annotation': base_path / 'annotations' / 'instances_val2017.json'
    }
}


with open(base_path/'categories.json') as f:
    category_synonyms = json.load(f)
    category_synonyms = {int(k):v for k,v in category_synonyms.items()}


train_tfms = A.Compose([
    A.Resize(height=640,width=640),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=15,
        p=0.3,
        border_mode=0
    ),
    A.AtLeastOneBBoxRandomCrop(height=512, width=512, erosion_factor=0.2, p=1.0),
    A.HorizontalFlip(p=0.5),

], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True, min_area=35000))


processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
processor.save_pretrained('./SmolVLM-256M-Detection2')

image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]


def collate_fn(examples,last_user_token='<end_of_utterance>'):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["image"] for example in examples]
    batch = processor(
        text=texts,
        images=images,
        return_tensors='pt',
        size={'longest_edge': 512},
        padding=True
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels

    # Train on Completion only
    last_user_token_id = processor.tokenizer.convert_tokens_to_ids(last_user_token)
    for batch_idx in range(batch['input_ids'].shape[0]):
        response_pos = torch.where(batch['input_ids'][batch_idx]==last_user_token_id)[0][0] # mask everything till User token
        batch['labels'][batch_idx, :response_pos+1] = -100

    return batch


train_ds = CocoDataset(
    root=ds_info['train']['dir'],
    annFile=ds_info['train']['annotation'],
    augments=train_tfms,
    synonyms=category_synonyms
)
print('total samples:',len(train_ds))

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
samples = next(iter(train_dl))
print({k:s.shape for k,s in samples.items()})
x = samples['input_ids'][0]
decoded_x = processor.tokenizer.decode(samples['input_ids'][0].numpy(),skip_special_tokens=False)
print(x)
print(decoded_x)
print(samples['labels'][0])
print(samples['attention_mask'][0])

DEVICE = 'cuda'


model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
).to(DEVICE)


for n,p in model.named_parameters():
    if 'attn' in n:
        p.requires_grad = True
    else:
        p.requires_grad = False

print(f"{(sum(p.numel() for p in model.parameters() if p.requires_grad)):,} / {(sum(p.numel() for p in model.parameters())):,}")


train_config = SFTConfig(
    output_dir='./SmolVLM-256M-Detection2',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.05,
    num_train_epochs=12,
    lr_scheduler_type='cosine',
    warmup_steps=3000,
    logging_steps=250,
    save_steps=1000,
    torch_empty_cache_steps=100,
    bf16=True,
    do_eval=False,
    push_to_hub=False,
    report_to='none',
    dataset_num_proc=4,
    save_total_limit=1,
    dataset_kwargs={'skip_prepare_dataset': True},
    dataset_text_field="",
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model,
    train_dataset=train_ds,
    data_collator=collate_fn,
    args=train_config,
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.model.save_pretrained('./SmolVLM-256M-Detection2')