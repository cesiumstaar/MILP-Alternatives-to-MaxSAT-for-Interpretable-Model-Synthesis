YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 43% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	56	1	0
0	0	43	1

MISCLASSIFICATION on test file: 17.3333% of 375 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	173	61	0
0	4	137	1

DECISION TREE (size=11)

F1 <= 0.100000 : 0 (31.0/1.4)
F1 > 0.100000 :
|   F10 <= 2.600000 :
|   |   F1 <= 0.980000 :
|   |   |   F10 <= 1.400000 : 0 (7.0/1.3)
|   |   |   F10 > 1.400000 :
|   |   |   |   F1 <= 0.410000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.410000 : 1 (7.0/1.3)
|   |   F1 > 0.980000 : 0 (15.0/1.3)
|   F10 > 2.600000 : 1 (37.0/2.6)
