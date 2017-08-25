Sentence Tokenizer
====================

## Basic Setting
* pip installation
    * `pip install -r requirement.txt`

## Train word2vec
`python w2v_train.py`
* You can train more with other data using model.train() function.

## Train
`python train.py --model small --data_path data/kr/kr.rd.tk`  
  * --tensorboard is optional and not implemented yet.

## Test
* Test sentences  
    `python test.py --model small --data_path data/kr/kr.rd.tk --test_sent "나는 다나입니다."`
* Test files  
    `python test.py --model small --data_path data/kr/kr.rd.tk --test_data_path ./data/result/test.kr`
  * Test config has to be same with train config.
  * --lan is optional and not implemented yet.
