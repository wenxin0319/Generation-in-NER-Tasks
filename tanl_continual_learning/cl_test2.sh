export DEBUG=1
export GOLD=1
export DROPOUT_PROB=0
export STRATEGY=ADD
export FEWSHOT_NUM=10
export CLASS_OPTION=1
export CLASS_FIRST=1
python3 run_cl2.py ace2005_ner_cl -v --gold
