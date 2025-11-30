YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 48.0769% of 52 cases. 

MISCLASSIFICATION on training: 3.84615% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	27	0	0
0	2	23	1

MISCLASSIFICATION on test file: 25% of 604 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	280	130	0
0	21	173	1

DECISION TREE (size=11)

F1 <= 0.100000 : 0 (20.0/1.3)
F1 > 0.100000 :
|   F10 <= 2.300000 :
|   |   F10 <= 1.700000 : 1 (2.0/1.0)
|   |   F10 > 1.700000 : 0 (7.0/3.4)
|   F10 > 2.300000 :
|   |   F1 <= 1.600000 : 1 (19.0/1.3)
|   |   F1 > 1.600000 :
|   |   |   F10 <= 4.000000 : 0 (2.0/1.0)
|   |   |   F10 > 4.000000 : 1 (2.0/1.0)
