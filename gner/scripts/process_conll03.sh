# Base Setting
# export RAW_DATA_PATH="./raw_data/CoNLL2003"
# export OUTPUT_PATH="./processed_data/conll03_bart/"

# mkdir $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b facebook/bart-large

# export OUTPUT_PATH="./processed_data/conll03_t5/"

# mkdir $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small

# Progressive Learning
# export RAW_DATA_PATH="./raw_data/CoNLL2003/2fold_82/Fold0"
# export OUTPUT_PATH="./processed_data/conll03_2fold_82_t5/fold_80"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small

# export RAW_DATA_PATH="./raw_data/CoNLL2003/2fold_82/Fold1"
# export OUTPUT_PATH="./processed_data/conll03_2fold_82_t5/fold_20"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small

# export RAW_DATA_PATH="./raw_data/CoNLL2003/2fold_91/Fold0"
# export OUTPUT_PATH="./processed_data/conll03_2fold_91_t5/fold_90"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small

# export RAW_DATA_PATH="./raw_data/CoNLL2003/2fold_91/Fold1"
# export OUTPUT_PATH="./processed_data/conll03_2fold_91_t5/fold_10"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small

# export RAW_DATA_PATH="./raw_data/CoNLL2003/2fold_955/Fold0"
# export OUTPUT_PATH="./processed_data/conll03_2fold_955_t5/fold_95"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small

# export RAW_DATA_PATH="./raw_data/CoNLL2003/2fold_955/Fold1"
# export OUTPUT_PATH="./processed_data/conll03_2fold_955_t5/fold_5"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small

# Interpolation
## 4 fold
# export RAW_DATA_PATH="./raw_data/CoNLL2003/4fold/Fold0"
# export OUTPUT_PATH="./processed_data/conll03_interpo_4split_t5/fold0"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types LOC MISC ORG

# export RAW_DATA_PATH="./raw_data/CoNLL2003/4fold/Fold1"
# export OUTPUT_PATH="./processed_data/conll03_interpo_4split_t5/fold1"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types LOC MISC PER

# export RAW_DATA_PATH="./raw_data/CoNLL2003/4fold/Fold2"
# export OUTPUT_PATH="./processed_data/conll03_interpo_4split_t5/fold2"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types LOC ORG PER

# export RAW_DATA_PATH="./raw_data/CoNLL2003/4fold/Fold3"
# export OUTPUT_PATH="./processed_data/conll03_interpo_4split_t5/fold3"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types MISC ORG PER

# export FINAL_OUTPUT_PATH="./processed_data/conll03_interpo_4split_t5"
# ### combine data
# python preprocessing/combine_dataset.py -i $FINAL_OUTPUT_PATH -n train.json -o $FINAL_OUTPUT_PATH
# python preprocessing/combine_dataset.py -i $FINAL_OUTPUT_PATH -n dev.json -o $FINAL_OUTPUT_PATH


# ## 6 fold
# export RAW_DATA_PATH="./raw_data/CoNLL2003/6fold/Fold0"
# export OUTPUT_PATH="./processed_data/conll03_interpo_6split_t5/fold0"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types LOC MISC

# export RAW_DATA_PATH="./raw_data/CoNLL2003/6fold/Fold1"
# export OUTPUT_PATH="./processed_data/conll03_interpo_6split_t5/fold1"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER ORG

# export RAW_DATA_PATH="./raw_data/CoNLL2003/6fold/Fold2"
# export OUTPUT_PATH="./processed_data/conll03_interpo_6split_t5/fold2"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types LOC ORG

# export RAW_DATA_PATH="./raw_data/CoNLL2003/6fold/Fold3"
# export OUTPUT_PATH="./processed_data/conll03_interpo_6split_t5/fold3"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER MISC

# export RAW_DATA_PATH="./raw_data/CoNLL2003/6fold/Fold4"
# export OUTPUT_PATH="./processed_data/conll03_interpo_6split_t5/fold4"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types LOC PER

# export RAW_DATA_PATH="./raw_data/CoNLL2003/6fold/Fold5"
# export OUTPUT_PATH="./processed_data/conll03_interpo_6split_t5/fold5"

# mkdir -p $OUTPUT_PATH
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types ORG MISC

# export FINAL_OUTPUT_PATH="./processed_data/conll03_interpo_6split_t5"
# ### combine data
# python preprocessing/combine_dataset.py -i $FINAL_OUTPUT_PATH -n train.json -o $FINAL_OUTPUT_PATH
# python preprocessing/combine_dataset.py -i $FINAL_OUTPUT_PATH -n dev.json -o $FINAL_OUTPUT_PATH

## interploation
export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group1/fold1"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group1/fold1"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER ORG LOC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group1/fold2"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group1/fold2"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types ORG LOC MISC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group1/fold3"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group1/fold3"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER ORG MISC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group1/"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group1_final"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER LOC MISC

### combine data
python preprocessing/combine_dataset.py -i ./processed_data/conll03_4group_dataset/group1 -n train.json -o ./processed_data/conll03_4group_dataset/group1_final

python preprocessing/combine_dataset.py -i ./processed_data/conll03_4group_dataset/group1 -n dev.json -o ./processed_data/conll03_4group_dataset/group1_final


export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group2/fold1"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group2/fold1"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types ORG LOC MISC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group2/fold2"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group2/fold2"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER ORG MISC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group2/fold3"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group2/fold3"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER LOC MISC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group2/"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group2_final"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER ORG LOC

### combine data
python preprocessing/combine_dataset.py -i ./processed_data/conll03_4group_dataset/group2 -n train.json -o ./processed_data/conll03_4group_dataset/group2_final

python preprocessing/combine_dataset.py -i ./processed_data/conll03_4group_dataset/group2 -n dev.json -o ./processed_data/conll03_4group_dataset/group2_final3


export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group3/fold1"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group3/fold1"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER ORG MISC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group3/fold2"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group3/fold2"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER LOC MISC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group3/fold3"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group3/fold3"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER ORG LOC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group3/"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group3_final"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types ORG LOC MISC

### combine data
python preprocessing/combine_dataset.py -i ./processed_data/conll03_4group_dataset/group3 -n train.json -o ./processed_data/conll03_4group_dataset/group3_final

python preprocessing/combine_dataset.py -i ./processed_data/conll03_4group_dataset/group3 -n dev.json -o ./processed_data/conll03_4group_dataset/group3_final

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group4/fold1"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group4/fold1"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER LOC MISC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group4/fold2"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group4/fold2"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER ORG LOC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group4/fold3"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group4/fold3"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types ORG LOC MISC

export RAW_DATA_PATH="./raw_data/CoNLL2003/4group_dataset/group4/"
export OUTPUT_PATH="./processed_data/conll03_4group_dataset/group4_final"

mkdir -p $OUTPUT_PATH
python preprocessing/process_conll03.py -i $RAW_DATA_PATH -o $OUTPUT_PATH -b t5-small --valid_types PER LOC MISC

### combine data
python preprocessing/combine_dataset.py -i ./processed_data/conll03_4group_dataset/group4 -n train.json -o ./processed_data/conll03_4group_dataset/group4_final

python preprocessing/combine_dataset.py -i ./processed_data/conll03_4group_dataset/group4 -n dev.json -o ./processed_data/conll03_4group_dataset/group4_final