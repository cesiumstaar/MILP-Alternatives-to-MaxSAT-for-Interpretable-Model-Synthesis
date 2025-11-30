YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45.2381% of 84 cases. 

MISCLASSIFICATION on training: 0% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	46	0	0
0	0	38	1

MISCLASSIFICATION on test file: 14.1141% of 333 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	185	38	0
0	9	101	1

DECISION TREE (size=11)

F1 <= -0.200000 : 0 (24.0/1.3)
F1 > -0.200000 :
|   F10 <= 2.500000 :
|   |   F1 <= 0.800000 :
|   |   |   F10 <= 1.300000 : 0 (6.0/1.2)
|   |   |   F10 > 1.300000 :
|   |   |   |   F1 <= 0.440000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.440000 : 1 (8.0/1.3)
|   |   F1 > 0.800000 : 0 (12.0/1.3)
|   F10 > 2.500000 : 1 (30.0/1.4)
