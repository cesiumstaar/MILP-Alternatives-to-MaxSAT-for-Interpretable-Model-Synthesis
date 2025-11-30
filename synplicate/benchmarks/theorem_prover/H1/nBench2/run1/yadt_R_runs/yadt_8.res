YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.1622% of 148 cases. 

MISCLASSIFICATION on training: 0% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	93	0	0
0	0	55	1

MISCLASSIFICATION on test file: 13.6456% of 982 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	560	121	0
0	13	288	1

DECISION TREE (size=15)

F1 <= -0.680000 : 0 (58.0/1.4)
F1 > -0.680000 :
|   F10 <= 2.700000 :
|   |   F10 <= 2.100000 : 0 (26.0/1.3)
|   |   F10 > 2.100000 :
|   |   |   F1 <= 1.310000 :
|   |   |   |   F1 <= 0.240000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.240000 : 1 (5.0/1.2)
|   |   |   F1 > 1.310000 : 0 (4.0/1.2)
|   F10 > 2.700000 :
|   |   F10 <= 3.300000 :
|   |   |   F1 <= 1.310000 : 1 (5.0/1.2)
|   |   |   F1 > 1.310000 : 0 (2.0/1.0)
|   |   F10 > 3.300000 : 1 (45.0/1.4)
