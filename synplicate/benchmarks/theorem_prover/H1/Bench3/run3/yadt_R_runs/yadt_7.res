YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 35.6061% of 132 cases. 

MISCLASSIFICATION on training: 0.757576% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	84	1	0
0	0	47	1

MISCLASSIFICATION on test file: 13.5524% of 487 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	273	60	0
0	6	148	1

DECISION TREE (size=15)

F1 <= -0.060000 : 0 (42.0/1.4)
F1 > -0.060000 :
|   F10 <= 2.700000 :
|   |   F1 <= 1.000000 :
|   |   |   F10 <= 1.300000 : 0 (11.0/1.3)
|   |   |   F10 > 1.300000 :
|   |   |   |   F1 <= 0.310000 : 0 (5.0/1.2)
|   |   |   |   F1 > 0.310000 : 1 (10.0/2.4)
|   |   F1 > 1.000000 : 0 (23.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 0.210000 :
|   |   |   F10 <= 3.500000 : 0 (3.0/1.1)
|   |   |   F10 > 3.500000 : 1 (2.0/1.0)
|   |   F1 > 0.210000 : 1 (36.0/1.4)
