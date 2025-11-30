YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.3913% of 115 cases. 

MISCLASSIFICATION on training: 0.869565% of 115 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	71	1	0
0	0	43	1

MISCLASSIFICATION on test file: 12.3724% of 784 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	471	82	0
0	15	216	1

DECISION TREE (size=13)

F1 <= -0.280000 : 0 (40.0/1.4)
F1 > -0.280000 :
|   F10 <= 2.600000 :
|   |   F1 <= 1.190000 :
|   |   |   F10 <= 1.200000 : 0 (8.0/1.3)
|   |   |   F10 > 1.200000 : 1 (10.0/2.4)
|   |   F1 > 1.190000 : 0 (19.0/1.3)
|   F10 > 2.600000 :
|   |   F1 <= 1.720000 : 1 (29.0/1.4)
|   |   F1 > 1.720000 :
|   |   |   F10 <= 3.700000 : 0 (4.0/1.2)
|   |   |   F10 > 3.700000 : 1 (5.0/1.2)
