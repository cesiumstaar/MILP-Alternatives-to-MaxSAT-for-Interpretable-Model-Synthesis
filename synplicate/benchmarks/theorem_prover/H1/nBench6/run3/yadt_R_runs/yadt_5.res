YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.3838% of 99 cases. 

MISCLASSIFICATION on training: 1.0101% of 99 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	60	1	0
0	0	38	1

MISCLASSIFICATION on test file: 13.1649% of 752 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	443	84	0
0	15	210	1

DECISION TREE (size=13)

F1 <= -0.280000 : 0 (38.0/1.4)
F1 > -0.280000 :
|   F10 <= 2.600000 :
|   |   F1 <= 1.190000 :
|   |   |   F10 <= 1.200000 : 0 (6.0/1.2)
|   |   |   F10 > 1.200000 : 1 (10.0/2.4)
|   |   F1 > 1.190000 : 0 (13.0/1.3)
|   F10 > 2.600000 :
|   |   F1 <= 1.720000 : 1 (25.0/1.3)
|   |   F1 > 1.720000 :
|   |   |   F10 <= 3.700000 : 0 (3.0/1.1)
|   |   |   F10 > 3.700000 : 1 (4.0/1.2)
