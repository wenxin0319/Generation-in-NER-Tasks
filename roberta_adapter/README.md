# Roberta

#### train stage1 dataset
```
python3 train1.py -c config/config_conll.json
```

#### evaluate stage1 dataset
```
python3 evaluate1.py -c config/config_conll.json -w output/conll/20210826_060047/best.role.mdl
```

#### train stage1,2 dataset(adding the adapter)
```
python3 train2.py -c config/config_conll.json
```

#### evaluate stage1,2 dataset(adding the adapter)
```
python3 evaluate2.py -c config/config_conll.json -w output/conll/20210826_060047/best.role.mdl
```

#### External Evaluator Single
```
python external_scorer_ner.py -p example.pred.test.json -g processed_data/conll03_t5/test.json --verbose
```

#### External Evaluator Ensemble
```
python ensemble_plus_scorer_ner.py -p 4model/4split_per_loc_org.json 4model/4split_org_misc_loc.json 4model/4split_per_loc_misc.json 4model/4split_per_org_misc.json -g processed_data/conll03_t5/test.json --verbose
```
