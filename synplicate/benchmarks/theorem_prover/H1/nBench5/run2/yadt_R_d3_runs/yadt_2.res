YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.4615% of 52 cases. 

MISCLASSIFICATION on training: 3.84615% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	18	2	0
0	0	32	1

MISCLASSIFICATION on test file: 29.3173% of 249 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	66	67	0
0	6	110	1

DECISION TREE (size=7)

F10 <= 1.500000 : 0 (9.0/1.3)
F10 > 1.500000 :
|   F1 <= 0.220000 :
|   |   F10 <= 5.000000 : 0 (9.0/1.3)
|   |   F10 > 5.000000 : 1 (2.0/1.0)
|   F1 > 0.220000 : 1 (32.0/3.8)
