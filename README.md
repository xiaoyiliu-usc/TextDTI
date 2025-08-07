# TEXT-DTI

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