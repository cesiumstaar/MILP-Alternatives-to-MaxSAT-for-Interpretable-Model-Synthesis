YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	60	1	0
0	0	39	1

MISCLASSIFICATION on test file: 20.1242% of 805 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	398	148	0
0	14	245	1

DECISION TREE (size=11)

F1 <= -0.550000 : 0 (38.0/1.4)
F1 > -0.550000 :
|   F10 <= 2.600000 :
|   |   F10 <= 1.600000 : 0 (15.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.240000 :
|   |   |   |   F1 <= 0.290000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.290000 : 1 (6.0/1.2)
|   |   |   F1 > 1.240000 : 0 (4.0/1.2)
|   F10 > 2.600000 : 1 (34.0/2.6)
