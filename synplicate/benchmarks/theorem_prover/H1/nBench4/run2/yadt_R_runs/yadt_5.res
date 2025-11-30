YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 35% of 100 cases. 

MISCLASSIFICATION on training: 3% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	64	1	0
0	2	33	1

MISCLASSIFICATION on test file: 13.1043% of 786 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	497	73	0
0	30	186	1

DECISION TREE (size=13)

F1 <= 0.160000 : 0 (40.0/2.6)
F1 > 0.160000 :
|   F10 <= 1.300000 : 0 (13.0/1.3)
|   F10 > 1.300000 :
|   |   F1 <= 1.630000 :
|   |   |   F10 <= 2.300000 :
|   |   |   |   F1 <= 0.970000 : 1 (7.0/2.4)
|   |   |   |   F1 > 0.970000 : 0 (2.0/1.0)
|   |   |   F10 > 2.300000 : 1 (24.0/1.3)
|   |   F1 > 1.630000 :
|   |   |   F10 <= 3.700000 : 0 (11.0/2.5)
|   |   |   F10 > 3.700000 : 1 (3.0/1.1)
