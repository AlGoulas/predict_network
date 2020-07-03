# predict_network
Predict missing connections in networks

# Description
The code builds predictive models that can estimate the existence and strength of edges in networks based on attributes of the nodes. Thus, predictions can be made for missing data, that is, connections, between pairs of nodes. In other words, the models predict the existence and significance (presence and strength) of interactions between objects, represented as edges and nodes, respectively.

The dataset that is included in this repo concerns biological neural networks but the methods are applicable to any type of networks.

Logistic regression is used to predict the existence of connections/edges and random forest regression for predicting the strength of connections/edges.

# Use
Create a virtual environment (e.g., with conda) with the specifications enlisted in requirements.txt. Download or clone this repository. If the virtual environment was created sucessfully, just run pred_conn.py. 

To do so, you have to specify the following path in the pred_conn.py file:
Folder to store the results (figures). E.g., 
```
path_results = Path("/Users/.../results")
```
You also have to specify what dataset from the full dataset shall be analyzed:
```
# 'macaque_monkey' 'Horvat_mouse' 'marmoset_monkey' macaque, mouse, and marmoset network data
dataset_name = 'macaque_monkey' 'Horvat_mouse' 'marmoset_monkey'
```
Note that the relative path to the dataset is specified correspondign to the path when downloading or cloning the repository. If for any reason you move or remane this folder then the approaproate path should be specified in the load_data function in the pred_conn.py file.  

# Data

The data are freely available from the following publication and the references publications and repositoires therein:
Goulas A, Majka P, Rosa MGP, Hilgetag CC (2019) A blueprint of mammalian cortical connectomes. PLoS Biol 17(3): e2005346. https://doi.org/10.1371/journal.pbio.2005346
