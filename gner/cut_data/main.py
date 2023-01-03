from process_txt import *
from process_json import *
from process_tokenizer import *
from process_type import *

#the original tokenizer for the processed/original file is T5-BASE

#task1
##-----process 8:2/9:1/95:5 split txt----
# name_list = ["processed/T5-base/train","processed/T5-base/dev","processed/T5-base/test"] 
# output_path="../raw_data/CoNLL2003/2fold_955"
# process_progressive_learn_not_drop_txt(name_list,output_path,0.95)

# output_path="../raw_data/CoNLL2003/2fold_91"
# process_progressive_learn_not_drop_txt(name_list,output_path,0.9)

# output_path="../raw_data/CoNLL2003/2fold_82"
# process_progressive_learn_not_drop_txt(name_list,output_path,0.8)

#task2
# #-----process remove labels partial three split txt----
# name_list = ["processed/T5-base/train","processed/T5-base/dev"]
# output_path="../raw_data/CoNLL2003/3fold"
# process_three_split_partial_txt(name_list,output_path)

#task3
# #-----process remove labels full four split txt----
# name_list = ["processed/T5-base/train","processed/T5-base/dev","processed/T5-base/test"]
# output_path="../raw_data/CoNLL2003/4fold"
# process_four_split_full_txt(name_list,output_path)

# # #-----process remove labels full six split txt----
# name_list = ["processed/T5-base/train","processed/T5-base/dev","processed/T5-base/test"]
# output_path="../raw_data/CoNLL2003/6fold"
# process_six_split_full_txt(name_list,output_path)


