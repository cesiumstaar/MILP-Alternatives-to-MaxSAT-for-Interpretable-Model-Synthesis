YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.9048% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	30	1	0
0	0	53	1

MISCLASSIFICATION on test file: 22.956% of 318 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	103	69	0
0	4	142	1

DECISION TREE (size=11)

F1 <= 0.220000 :
|   F10 <= 4.800000 : 0 (21.0/1.3)
|   F10 > 4.800000 : 1 (5.0/2.3)
F1 > 0.220000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (6.0/1.2)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 1.000000 : 1 (5.0/1.2)
|   |   |   F1 > 1.000000 : 0 (3.0/1.1)
|   F10 > 2.500000 : 1 (44.0/1.4)
