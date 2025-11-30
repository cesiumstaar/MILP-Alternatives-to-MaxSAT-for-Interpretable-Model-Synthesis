YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 22.619% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	64	1	0
0	0	19	1

MISCLASSIFICATION on test file: 15.3324% of 737 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	477	49	0
0	64	147	1

DECISION TREE (size=13)

F10 <= 1.900000 : 0 (39.0/1.4)
F10 > 1.900000 :
|   F1 <= 0.270000 :
|   |   F10 <= 4.300000 : 0 (18.0/1.3)
|   |   F10 > 4.300000 : 1 (3.0/2.1)
|   F1 > 0.270000 :
|   |   F1 <= 0.950000 : 1 (15.0/1.3)
|   |   F1 > 0.950000 :
|   |   |   F10 <= 2.500000 : 0 (5.0/1.2)
|   |   |   F10 > 2.500000 :
|   |   |   |   F1 <= 1.840000 : 1 (2.0/1.0)
|   |   |   |   F1 > 1.840000 : 0 (2.0/1.0)
