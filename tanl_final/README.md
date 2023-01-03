# modifyed_class_op_tanl

modifying config.ini to change from loading t5-base to your own path of model
``bash run.sh `

Using ``python3 run.py conll03`` or ``python3 run.py ontonotes`` or ``python3 run.py multi_dataset_ner`` or ``python3 run.py multitask_ner_finetune_conll03`` or ``python3 run.py multitask_ner_finetune_ontonotes`` to run different tasks

#### installations and setups

please run ``pip3 install -r requirements.txt --user``, python3.6 - python3.8 is always fine, and if needing debug, ipdb is used and run ``pip3 install ipdb --user`` to install that

#### hyperparameters

CLASS_OPTION: ''' whether to add class option in the input of tanl '''

``CLASS_OPTION=1`` means adding class option and  ``CLASS_OPTION=0`` means not adding class option

CLASS_FIRST: ''' whether to make class option first before input sentence or make class option after the sentence '''

``CLASS_FIRST=1`` means adding class first and ``CLASS_FIRST=0`` means adding class in the last (not first)

DEBUG: ''' whether to print debug information '''

``DEBUG=1`` means to print debug information and ``DEBUG=0`` means not printing debug information

<!-- DROPOUT_PROB: ''' the probability to dropout classes '''

``DROPOUT_PROB=0.2`` menas each class have a probability of 0.2 to be dropped out -->

<!-- os.environ: 
1. FEWSHOT_NUM: this means the particular number you want a few shot label to add in the training data per epoch. 
For example, if ``FEWSHOT_NUM`` is 5, the examples per epoch would be 5.
2. STRATEGY: this means the strategy of training you choose. You must fill in this by assigning at least ``export STRATEGY=ADD``, by this you will enter add mode, which corresponds to normal adding and removing experiments with dropout rate denoted with ``DROPOUT_PROB``. 

    If you choose ``export STRATEGY==HOLD``, you will get what Prof. nanyun says, to hold the labels appearing in the gold entities and to randomly add other labels, the control of randomness is by ``ACCEPT_PROB``, ``export ACCEPT_PROB=0.2`` means each other class has a probability of 0.2 to be added. 

    If you choose ``export STRATEGY==FEWSHOT``, you will get fewshot version as 0805 I Hung says, retaining static classes without dropout and add few shot samples, worthnoting, this is the same with the ``export STRATEGY=ADD; export DROPOUT_PROB=0.0`` situation.
3. ACCEPT_PROB: see above. -->

#### Outcome

output and input in detail are in logs like ``output1_veh.log`` and ``output_weapon.log`` whose names indicate their function: those logs are testing result removing that particular class (trained on random dropout with dropout probability 0.3, if there's any question, refer to the above)

output gold-answer counting numbers are in ``gold-answer.log`` with these classes removing

output acc are in ``without_answer.log``

##### Randomness

random.shuffle()

details in ``input_formats.py``

##### How to begin

see ``run.sh`` class option are in ``config.ini`` with two keys: ``mapping_input_choice`` and ``mapping_output_choice``.

Implementation: basic choice is  ``static_class_choice``, 
<!-- when you want to add fewshot class, you write it in ``add_class_choice``, now all the classes are added with the same times. ``FEWSHOT_NUM`` controls the number.  -->

#### notes
we are now not supporting experiments like what I-hung has done to continual the training for different tasks(human operation)
but we will take time to modify this thing.