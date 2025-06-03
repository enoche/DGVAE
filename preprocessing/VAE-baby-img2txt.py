
# load images
import clip
import torch
import re

import pandas as pd
topk_txt = 20

device = "cuda:2" if torch.cuda.is_available() else "cpu"
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
model, preprocess = clip.load('ViT-B/32', device)

text_inputs = torch.cat([clip.tokenize(c) for c in item_words]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
# test encoded
print('text encoded')
text_features /= text_features.norm(dim=-1, keepdim=True)
text_features = text_features.T

item_num = df.shape[0]
#item_num = 100
print('# of items', item_num)

img_txt_ls = []
item_no_imgs = []
for i in tqdm(range(item_num)):
    fl_name = join(img_dir, f'{i}.jpg')
    if isfile(fl_name):
        #print(fl_name)
        image = Image.open(fl_name)
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        # Pick the top 5 most similar labels for the image
        del image_input
        torch.cuda.empty_cache()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features).softmax(dim=-1)
        del image_features
        torch.cuda.empty_cache()
        values, indices = similarity[0].topk(topk_txt)
        # Print the result
        #print("\nTop predictions:\n")
        tmp_text = ''
        for value, index in zip(values, indices):
            #print(f"{index}-{item_words[index]:>16s}: {100 * value.item():.2f}%")
            tmp_text = tmp_text + ' ' + item_words[index]
        img_txt_ls.append(tmp_text)
    else:
        item_no_imgs.append(i)
        img_txt_ls.append('')

print('# of images with text', len(img_txt_ls)-len(item_no_imgs))
print('# of images without text', len(item_no_imgs))

# save

df = pd.DataFrame(img_txt_ls, columns=["img2txt"])
df.to_csv(f'{dataset}-img2txt{topk_txt}.csv', index=False)

df1 = pd.DataFrame(item_no_imgs, columns=["imgwotxt"])
df1.to_csv(f'{dataset}-imgwotxt.csv', index=False)


