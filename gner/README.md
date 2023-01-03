
## Datasets
We support `conll03`

### Preprocessing
#### `conll03`
1. Prepare data and Put the processed data into the folder `processed_data/`
3. Run `./scripts/process_conll03.sh`

## Training
#### `conll03`
Run `./scripts/train_conll03.sh`

The model will be stored at `./output/conll03/[timestamp]/best_model.mdl` in default.

## Evaluation
#### `conll03`
```Bash
python eval_ner_conll03.py -cner config/config_ner_conll03.json -ner [ner_model]
```

## Testing Environment
- CUDA 10.2
- Python==3.8.10
- PyTorch==1.8.1
- torchvision==0.9.1
- tqdm==4.61.0
- tensorboardx==2.2
- transformers==4.6.1
- lxml==4.6.3
- beautifulsoup4==4.9.3
- bs4==0.0.1
- nltk==3.6.2
- jieba==0.42.1
- stanza==1.2
- sentencepiece==0.1.95
- ipdb==0.13.9