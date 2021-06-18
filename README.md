# FakeNewsDetection
Fake News Detection on Twitter using Graph Deep Learning


Requirements for running the ``CreateDataset`` code:
```shell script
conda install python-graphviz
pip install searchtweets-v2
pip install TwitterAPI

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-geometric
```
where ```${CUDA}``` and ```${TORCH}``` should be replaced by your specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111) and PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.8.0),

To create the word embeddings it is required to create a subdir in ``CreateData`` called ``resources``, in this folder you need to place the ``glove.twitter.27B.200d.txt`` file.
You can download the file here: http://nlp.stanford.edu/data/glove.twitter.27B.zip