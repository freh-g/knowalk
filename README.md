# KnoWalk

Knowalk is an algorithm that produces embeddings out of a knowledge graph. The knowledge graph has to be fed to the algorithm in a .csv format without index  and with specific headers i.e. 

- source: id of the source
- target: id of the target
- rel_type: type of the interaction
- source_type: type of the source
- target_type: type of the target

An example i given in kg_edgelist.csv file. In this knowledge graph we have ~60 types of relationship and 4 node types (functions, phenotypes, drugs, proteins).

KnoWalk need several dependencies, an environment named KnoWalk can be created by running

```
conda env create -f KnoWalk.yml
```

The algorithm builds a nx.MultiDiGraph() from the edgelist file and builds biased walks that are used as input to Word2Vec algorithms.

For tuning the algorithm the script accepts several parameters that can be listed by running

```
python3 KW2VEC.py -h
```

An important component is the weight dictionary. It have to be passed during the call in a form of python dictionary in which the keys are tuples reporting the jump from node type to node type to weight (node types have to be the same in source_type and target_type columns of the edgelist) and the keys is the assigned value. For the kg_edgelist.csv a correct call specifing weights could be:

```

python KW2VEC.py -e kg_edgelist.csv -w "{('drug','protein'): 0,('protein','function'): 10,('function','phenotype'):100}" -s True -o KnoWalk_embeddings.pickle
```







