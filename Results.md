# SimpleRNN

## Word based
| Vocab Size | Sentence Length | Hidden Size | Batch Size | Learning Rate | Scheduler | Optimizer | Epochs | Tr Acc | Va Acc | Tr CE   | Va CE   | Initialization | NonLinearity |
| ---------- | --------------- | ----------- | ---------- | ------------- | --------- | --------- | ------ | ------ | ------ | ------- | ------- | -------------- | ------------ |
| 10000      | 10              | 8           | 2048       | 1e0           | -         | SGD,N,M=.9| 25     | 16.03  | 16.04  | 4.4626  | 4.4634  | -              | tanh         |
| 10000      | 10              | 8           | 128        | 1e0           | -         | SGD,N,M=.9| 25     | 15.83  | 15.79  | 4.4577  | 4.4595  | -              | tanh         |
| 10000      | 10              | 128         | 128        | 1e0           | -         | SGD,N,M=.9| 25     | 16.14  | 16.11  | 4.3888  | 4.3937  | -              | tanh         |
| 10000      | 10              | 128         | 128        | 1e0           | Plat,P=2  | SGD,N,M=.9| 25     | 17.57  | 17.56  | 4.3140  | 4.3197  | -              | tanh         |
| 10000      | 5               | 128         | 128        | 1e0           | Plat,P=2  | SGD,N,M=.9| 25     | 16.75  | 16.79  | 4.3655  | 4.3679  | -              | tanh         |
| 10000      | 20              | 128         | 128        | 1e0           | Plat,P=2  | SGD,N,M=.9| 25     | 17.47  | 17.45  | 4.3218  | 4.3265  | -              | tanh         |
| 10000      | 20              | 128         | 128        | 1e-1          | Plat,P=2  | SGD,N,M=.9| 25     | 17.79  | 17.74  | 4.3064  | 4.3106  | Xavier Uniform | tanh         |
| 10000      | 20              | 128         | 128        | 1e-1          | Plat,P=2  | SGD,N,M=.9| 25     | 18.03  | 18.00  | 4.2796  | 4.2851  | Xavier Normal  | tanh         |
| 10000      | 20              | 128         | 128        | 1e-2          | Plat,P=2  | SGD,N,M=.9| 25     | 17.65  | 17.64  | 4.3156  | 4.3185  | Kaiming Uniform| relu         |
| 10000      | 20              | 128         | 128        | 1e-2          | Plat,P=2  | SGD,N,M=.9| 25     | 17.37  | 17.36  | 4.3401  | 4.3418  | Kaiming Normal | relu         |
| 10000      | 20              | 128         | 128        | 1e-2          | Plat,P=2  | SGD,N,M=.9| 25     | 17.07  | 17.06  | 4.3667  | 4.3686  | KN+ID          | relu         |
| 10000      | 20              | 128         | 128        | 1e-3          | Plat,P=2  | Adam      | 25     | 18.84  | 18.81  | 4.2198  | 4.2285  | KN+ID          | relu         |

## Character Based
| Sentence Length | Embedding Size | Hidden Size | Batch Size | Learning Rate | Scheduler | Optimizer | Epochs | Tr Acc | Va Acc | Tr CE   | Va CE   | Initialization | NonLinearity |
| --------------- | -------------- | ----------- | ---------- | ------------- | --------- | --------- | ------ | ------ | ------ | ------- | ------- | -------------- | ------------ |
| 100             | 128            | 128         | 128        | 1e-1          | Plat,P=2  | SGD,N,M=.9| 25     | 54.02  | 54.01  | 1.5212  | 1.5216  | Xavier Normal  | tanh         |
| 100             | 128            | 128         | 128        | 1e-2          | Plat,P=2  | SGD,N,M=.9| 25     | 49.94  | 49.94  | 1.6740  | 1.6742  | Kaiming Uniform| relu         |
| 100             | 128            | 128         | 128        | 1e-2          | Plat,P=2  | SGD,N,M=.9| 25     | 51.41  | 51.39  | 1.6220  | 1.6220  | Kaiming Normal | relu         |
| 100             | 128            | 128         | 128        | 1e-2          | Plat,P=2  | SGD,N,M=.9| 25     | 47.07  | 47.05  | 1.7677  | 1.7684  | KN+ID          | relu         |
| 100             | 128            | 128         | 128        | 1e-3          | Plat,P=2  | Adam      | 25     | 54.39  | 54.37  | 1.5035  | 1.5040  | KN+ID          | relu         |
| 100             | 2              | 2           | 128        | 1e-3          | Plat,P=2  | Adam      | 25     | 18.33  | 18.35  | 2.6696  | 2.6696  | KN+ID          | relu         |
| 100             | 4              | 4           | 128        | 1e-3          | Plat,P=2  | Adam      | 25     | 27.25  | 27.23  | 2.4145  | 2.4154  | KN+ID          | relu         |
| 100             | 4              | 128         | 128        | 1e-3          | Plat,P=2  | Adam      | 25     | 52.20  | 52.20  | 1.5887  | 1.5893  | KN+ID          | relu         |
| 100             | 8              | 128         | 128        | 1e-3          | Plat,P=2  | Adam      | 25     | 53.70  | 53.68  | 1.5328  | 1.5331  | KN+ID          | relu         |
