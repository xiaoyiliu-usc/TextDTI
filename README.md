# TEXT-DTI

## Abstact
Predicting drug–target interactions (DTIs) is a crucial task in drug discovery. Recent advances in deep learning, particularly the application of large language models (LLMs), have shown promise in encoding sequential information from SMILES strings and protein sequences. However, integrating these diverse modalities remains a challenge. In this paper, we propose TextDTI, a multi-modal framework that simultaneously exploits sequential and structural representations. First, TextDTI utilizes Pretrained Language Models (PLMs) to generate functional description texts for proteins represented by their amino acid sequences. Next, the generated texts and the SMILES sequences of drugs are encoded into corresponding feature representations by other separate LLMs. Third, drug and target characteristics are fused through convolutional and graph-based modules. Finally, unidentified drug-target interactions are classified using a multilayer perceptron neural network. We further enhance feature alignment through adversarial learning and contrastive loss, resulting in robust and high-performance DTI prediction. Experiments conducted on multiple datasets in both single-domain and cross-domain settings demonstrate that our model outperforms other baseline methods. 

![image](./img/Fig 1.pfd)


The code has been tested in the following environment:
Python-3.8,
PyTorch-1.8.0,
CUDA-11.1,

Dataset link: https://drive.google.com/drive/folders/1rwOJQXslJEYtNKHm9LbXBxiQBvbSsC67?usp=sharing

Data Process:

The original data is first processed by the code document "predata.py";

Then we can use "data_split.py" to split the data sets; 
              
After that, we can run "main.py".

The only thing we have to do is to write the data protocols and the save path;
              
```python
>>> data_select = "B_D_H_to_C"
>>> setting = "B_D_H_to_C"
>>> file_model = './/best_model//' + setting
```

The specific data protocols are described in the function "def data_load(data_select, device)" in "main.py".
