YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 50% of 84 cases. 

MISCLASSIFICATION on training: 2.38095% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	41	1	0
0	1	41	1

MISCLASSIFICATION on test file: 18.5944% of 683 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	355	97	0
0	30	201	1

DECISION TREE (size=11)

F1 <= -0.220000 : 0 (30.0/1.4)
F1 > -0.220000 :
|   F10 <= 1.300000 : 0 (5.0/1.2)
|   F10 > 1.300000 :
|   |   F1 <= 1.210000 : 1 (35.0/2.6)
|   |   F1 > 1.210000 :
|   |   |   F10 <= 3.000000 : 0 (4.0/1.2)
|   |   |   F10 > 3.000000 :
|   |   |   |   F1 <= 2.210000 : 1 (7.0/1.3)
|   |   |   |   F1 > 2.210000 : 0 (3.0/2.1)
