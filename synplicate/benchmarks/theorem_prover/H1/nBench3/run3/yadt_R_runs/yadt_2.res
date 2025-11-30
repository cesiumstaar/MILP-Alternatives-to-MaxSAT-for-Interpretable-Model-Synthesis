YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.1538% of 52 cases. 

MISCLASSIFICATION on training: 1.92308% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	23	1	0
0	0	28	1

MISCLASSIFICATION on test file: 13.4615% of 260 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	120	32	0
0	3	105	1

DECISION TREE (size=9)

F1 <= 0.110000 : 0 (13.0/1.3)
F1 > 0.110000 :
|   F10 <= 1.300000 : 0 (8.0/1.3)
|   F10 > 1.300000 :
|   |   F10 <= 2.600000 :
|   |   |   F1 <= 1.230000 : 1 (6.0/2.3)
|   |   |   F1 > 1.230000 : 0 (2.0/1.0)
|   |   F10 > 2.600000 : 1 (23.0/1.3)
