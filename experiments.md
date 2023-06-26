# EXPERIMENT

## Datasets and Baseline
### Datasets

- Airports:     
  - nodes: 1190,  edges: 13599,  avg(degree): 22.86
- Wiki:         
  - nodes: 2405,  edges: 17981,  avg(degree): 13.74
- Facebook:      
  - nodes: 4039,  edges: 88234,  avg(degree): 43.69
- Actors:        
  - nodes: 7600,  edges: 30019,  avg(degree): 07.90
- Github:        
  - nodes: 37700, edges: 578006, avg(degree): 30.66

### Methods

- GCNR
- GCNSI
- NetSleuth

## Settings and Optimizations
### Propagation Models

- SI: GCNR, GCNSI, Netsleuth
- SIR: GCNR, GCNSI

Parameters: 

"As shown in many existing work [34, 42], the infection probability p in SI and SIR is sampled from 
uniform distribution U (0, 1) and the recovery probability q in SIR is sampled from uniform distribution U (0, p).", 

"In addition, we restrict the lowest infection rate of G as 30% (as the same in [34]) and this setting is the same for all datasets."

### Runs

Runs: All the introduced results are under over 500 independent runs to ensure the credibility (as the same in [26, 34]).

### Sources

Numbers of sources:
- Airports, Wiki: 3, 5
- Facebook, Actors: 8, 10
- Github: 20, 24

### Evaluation Metrics

- Rank
- Min-Matching-Distance
- AUC-Score
- TP/FP
- (F1?)

### Hyper-parameters

- learning rate
- GCN layers
- hidden size
- batch size
- epoch

## Results

Evaluate Metrics, different graphics

## Impact of Parameters

- learning rate
- GCN layers
- hidden size

Validation of the parameter settings on a case study (one data set)

"Therefore, when we adjust a parameter, the other three are fixed with the best settings, and the impact of each 
parameter is introduced separately in the rest of this section."

## Scalability

Complexity of the models

## Discussion
Comparing models