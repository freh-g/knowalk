# KnoWalk

Knowalk is an algorithm that produces embeddings out of a knowledge graph. The knowledge graph has to be fed to the algorithm in a .csv format without index  and with specific headers i.e. 

- source: id of the source
- target: id of the target
- rel_type: type of the interaction
- source_type: type of the source
- target_type: type of the target

As an example is available in data/kg_edgelist.csv. This knowledge graph is composed of 41k nodes of 4 types (functions, phenotype, drug and protein) and ~60 types of relationships.

KnoWalk need several dependencies, an environment named KnoWalk can be created by running

```
conda env create -f KnoWalk.yml
```
Then activate the environment with 


```
conda activate KnoWalk
```



The algorithm builds a nx.MultiDiGraph() from the edgelist file and builds biased walks that are used as input to Word2Vec algorithms.

For tuning the algorithm the script accepts several parameters that can be listed by running

```
python3 KW2VEC.py -h
```

An important component is the weight dictionary. It have to be passed during the call in a form of python dictionary in which the keys are tuples reporting the jump from node type to node type to weight (node types have to be the same in source_type and target_type columns of the edgelist) and the keys is the assigned value. For the kg_edgelist.csv a correct call specifing weights could be:

```
python KW2VEC.py -e data/kg_edgelist.csv -w "{('drug','protein'): 0,('protein','function'): 10,('function','phenotype'):100}" -s True -o outputs/kg_embeddings.pickle
```







