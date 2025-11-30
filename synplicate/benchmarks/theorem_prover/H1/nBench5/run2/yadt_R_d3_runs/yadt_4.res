YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.9048% of 84 cases. 

MISCLASSIFICATION on training: 4.7619% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	27	4	0
0	0	53	1

MISCLASSIFICATION on test file: 27.3585% of 318 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	89	83	0
0	4	142	1

DECISION TREE (size=9)

F1 <= 0.220000 :
|   F10 <= 4.800000 : 0 (21.0/1.3)
|   F10 > 4.800000 : 1 (5.0/2.3)
F1 > 0.220000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (6.0/1.2)
|   |   F10 > 1.500000 : 1 (8.0/4.5)
|   F10 > 2.500000 : 1 (44.0/1.4)
