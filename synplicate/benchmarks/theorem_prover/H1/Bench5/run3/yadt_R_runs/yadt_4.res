YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45.2381% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	45	1	0
0	0	38	1

MISCLASSIFICATION on test file: 17.0659% of 334 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	165	52	0
0	5	112	1

DECISION TREE (size=11)

F1 <= 0.270000 :
|   F10 <= 4.600000 : 0 (28.0/1.4)
|   F10 > 4.600000 : 1 (3.0/2.1)
F1 > 0.270000 :
|   F10 <= 1.400000 : 0 (11.0/1.3)
|   F10 > 1.400000 :
|   |   F1 <= 1.320000 : 1 (28.0/1.4)
|   |   F1 > 1.320000 :
|   |   |   F10 <= 2.900000 : 0 (6.0/1.2)
|   |   |   F10 > 2.900000 : 1 (8.0/1.3)
