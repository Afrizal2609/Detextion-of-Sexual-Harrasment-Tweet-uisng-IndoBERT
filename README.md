# Detection of Sexual Harrasment Tweet on Twitter Using Machine Learning
Hi guys, this repository contains my final project in college. I conducted research to compare sexual harassment tweets using the state of the art model in the field of NLP. I used IndoBERT, IndoXLNet, RoBERTa, BERT, and XLNet models. The pretrained models that I used can be seen in the table below.

|        Model        |                         Source                          |
| :-----------------: | :-----------------------------------------------------: | 
|  bert-base-uncased  |         https://huggingface.co/bert-base-uncased        |  
|    RoBERTa-base     |           https://huggingface.co/roberta-base           |     
|  indobert-base-p2   |  https://huggingface.co/indobenchmark/indobert-base-p2  | 
|  xlnet-base-cased   |          https://huggingface.co/xlnet-base-cased        | 
|     indoxlnet       |          https://huggingface.co/tepanee/indoxlnet       | 


By using a dataset of 30,189 data divided into three parts: training data, validation data, and testing data with a ratio of 7:3:1, as well as batch size 16 and epoch 5, IndoBERT model produces the best performance with accuracy, f1-score, recall, and precision of 0.8587, 0.8646, 0.8897, and 0.8409, respectively. The IndoBERT model successfully classified correctly 1,350 sexual harassment data out of a total of 1,350. sexual harassment data from a total of 1,513 sexual harassment data in the testing data.

The implemented model can be accessed at the following link:
https://huggingface.co/spaces/itsam26/app-sexual-harassment