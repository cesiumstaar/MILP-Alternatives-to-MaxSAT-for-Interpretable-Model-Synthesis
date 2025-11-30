YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45.122% of 164 cases. 

MISCLASSIFICATION on training: 0.609756% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	74	0	0
0	1	89	1

MISCLASSIFICATION on test file: 20.3327% of 541 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	196	92	0
0	18	235	1

DECISION TREE (size=13)

F1 <= 0.000000 :
|   F10 <= 4.500000 : 0 (38.0/1.4)
|   F10 > 4.500000 :
|   |   F1 <= -0.160000 : 0 (13.0/2.5)
|   |   F1 > -0.160000 : 1 (3.0/1.1)
F1 > 0.000000 :
|   F10 <= 3.000000 :
|   |   F1 <= 1.330000 :
|   |   |   F10 <= 1.600000 : 0 (7.0/1.3)
|   |   |   F10 > 1.600000 : 1 (12.0/1.3)
|   |   F1 > 1.330000 : 0 (17.0/1.3)
|   F10 > 3.000000 : 1 (74.0/1.4)
