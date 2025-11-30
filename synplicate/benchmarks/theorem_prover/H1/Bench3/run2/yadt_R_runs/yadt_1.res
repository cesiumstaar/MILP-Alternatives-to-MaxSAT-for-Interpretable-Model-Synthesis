YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 30.5556% of 36 cases. 

MISCLASSIFICATION on training: 2.77778% of 36 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	10	1	0
0	0	25	1

MISCLASSIFICATION on test file: 35.9091% of 220 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	47	76	0
0	3	94	1

DECISION TREE (size=7)

F1 <= 0.340000 :
|   F1 <= 0.100000 : 0 (7.0/1.3)
|   F1 > 0.100000 :
|   |   F10 <= 2.900000 : 0 (3.0/1.1)
|   |   F10 > 2.900000 : 1 (3.0/1.1)
F1 > 0.340000 : 1 (23.0/2.5)
