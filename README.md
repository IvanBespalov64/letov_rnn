## Letov RNN

This is the batch of scripts that gives you ability to feel like one of the most impressive soviet pank artists, Egor Letov. Based on GRU DL model and trained on whole collection of his poems, this project can generate texts, based on learned. 
To just generate run:

```bash
python generate.py --model model.pkl --length <number of words> --prefix <some prefix, maybe empty>
```

If you want to train model, run:

```bash
python train.py --input-dir data/poems.txt --model model.pkl
```

Requirements: 

- torch 
- pickle
- numpy
