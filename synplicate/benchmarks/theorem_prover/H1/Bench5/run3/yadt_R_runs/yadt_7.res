YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.2121% of 132 cases. 

MISCLASSIFICATION on training: 0.757576% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	70	1	0
0	0	61	1

MISCLASSIFICATION on test file: 18.4615% of 455 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	209	78	0
0	6	162	1

DECISION TREE (size=15)

F1 <= -0.030000 :
|   F10 <= 4.900000 : 0 (37.0/1.4)
|   F10 > 4.900000 :
|   |   F1 <= -0.370000 : 0 (3.0/1.1)
|   |   F1 > -0.370000 : 1 (2.0/1.0)
F1 > -0.030000 :
|   F10 <= 2.400000 :
|   |   F10 <= 1.400000 : 0 (20.0/1.3)
|   |   F10 > 1.400000 :
|   |   |   F1 <= 1.240000 :
|   |   |   |   F1 <= 0.320000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.320000 : 1 (10.0/1.3)
|   |   |   F1 > 1.240000 : 0 (8.0/1.3)
|   F10 > 2.400000 : 1 (50.0/2.6)
