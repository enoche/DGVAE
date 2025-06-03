
# load images
import clip
import torch
import re

# https://blog.shikoan.com/coca-blip2/

import pandas as pd
import os
import glob
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm
import numpy as np

topk_txt = 20

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset = 'baby14'
txt_file = f'meta-{dataset}.csv'
df = pd.read_csv(txt_file)
i_id, desc_str,title,brand, category = 'itemID', 'description','title','brand','categories'
df.sort_values(by=[i_id], inplace=True)
print('data loaded.', df.shape)

df['title'] = df['title'].fillna(" ")
df['description'] = df['description'].fillna(" ")
#df['ingredients'] = df['ingredients'].map(lambda x: x.replace('^',','))
df['brand'] = df['brand'].fillna(" ")
df['categories'] = df['categories'].fillna(" ")
sentences = ''

for i, row in df.iterrows():
    #sen = row['title'] + ' ' + row['brand'] + ' '
    sen = row['title'] + ' '
    cates = eval(row['categories'])
    if isinstance(cates, list):
        for c in cates[0]:
            sen = sen + c + ' '
    #sen += row[desc_str]
    sen = sen.replace('\n', ' ')

    sentences += sen

sentences = re.sub(r'[^\w\s]', ' ', sentences.lower())
# filtering
item_words = set()
for s in sentences.split():
    for c in s:
        has_digit = False
        if c.isdigit():
            has_digit = True
            break
    if not has_digit and len(s) >= 2 and s not in item_words:
        item_words.add(s)

item_words = sorted(list(item_words))
print('First 20 words', item_words[:20])
#sentences = ' '.join(s for s in sentences.split() if not any(c.isdigit() for c in s))
#item_words = list(set(sentences.split()))
print('# of unique words', len(item_words))
#exit(0)

from PIL import Image

from os import listdir
from os.path import isfile, join
from tqdm import tqdm

img_dir = 'raw-img'

onlyfiles = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
sorted_files = sorted(onlyfiles)

# Load the model
#device = "cpu"
#model, preprocess = clip.load('ViT-B/32', device)

device = "cuda:1"
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2", model_type="pretrain", is_eval=True, device=device)
model.visual_encoder.float()

text_embeddings = []
with torch.no_grad():
    for item in tqdm(item_words):
        text = model.tokenizer(
            item,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        text_feat = model.forward_text(text)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeddings.append(text_embed.mean(dim=0, keepdim=True))
    text_embeddings = torch.cat(text_embeddings, dim=0)
    print(text_embeddings.shape) # torch.Size([1000, 256])

print('text encoded')
text_features = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
text_features = text_features.T

item_num = df.shape[0]
#item_num = 100
print('# of items', item_num)

img_txt_ls = []
item_no_imgs = []
with torch.no_grad():
    for i in tqdm(range(item_num)):
        fl_name = join(img_dir, f'{i}.jpg')
        if isfile(fl_name):
            #print(fl_name)
            raw_image = Image.open(fl_name).convert("RGB")
            image_input = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            image_feat, vit_feat = model.forward_image(image_input)
            image_embed = model.vision_proj(image_feat)
            image_features = image_embed
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features).softmax(dim=-1)
            values, indices = similarity[0].topk(topk_txt)
            # Print the result
            # print("\nTop predictions:\n")
            tmp_text = ''
            for value, index in zip(values, indices):
                # print(f"{index}-{item_words[index]:>16s}: {100 * value.item():.2f}%")
                tmp_text = tmp_text + ' ' + item_words[index]
            img_txt_ls.append(tmp_text)
        else:
            item_no_imgs.append(i)
            img_txt_ls.append('')

print('# of images with text', len(img_txt_ls)-len(item_no_imgs))
print('# of images without text', len(item_no_imgs))

# save

df = pd.DataFrame(img_txt_ls, columns=["img2txt"])
df.to_csv(f'{dataset}-img2txt{topk_txt}-blip.csv', index=False)

df1 = pd.DataFrame(item_no_imgs, columns=["imgwotxt"])
df1.to_csv(f'{dataset}-imgwotxt-blip.csv', index=False)


