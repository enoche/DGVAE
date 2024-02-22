## Datasets

#### Pre-processed for quick test
- [Baby on GDrive](https://drive.google.com/drive/folders/1eCvCmemwKBYNRHT3e0iPZCf2WM1v2Abz?usp=sharing)

#### Pre-processing `baby` step-by-step
1. Multimodal feature extraction --> [MMRec](https://github.com/enoche/MMRec/tree/master/preprocessing)  
<Generated files: :  `baby.inter`, `image_feat.npy`, `text_feat.npy`>
2. Image --> Text mapping  with `CLIP` or `BLIP-2`
3. Constructing word vocabulary with `preprocessing/word-vocab.ipynb`  
<Generated files: `baby.train.ratings`, `baby.valid.ratings`, `baby.test.ratings`, `baby.uid.npy`, 
`baby.iid.npy`, `baby.vocab`, `baby.user.words.tfidf.npz`>
4. Move all data into `data/baby/`
5. DONE!
