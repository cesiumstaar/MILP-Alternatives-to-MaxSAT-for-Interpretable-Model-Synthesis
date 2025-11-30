YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 30.303% of 132 cases. 

MISCLASSIFICATION on training: 0.757576% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	92	0	0
0	1	39	1

MISCLASSIFICATION on test file: 11.6631% of 926 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	589	80	0
0	28	229	1

DECISION TREE (size=19)

F1 <= -0.130000 : 0 (50.0/1.4)
F1 > -0.130000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.400000 : 0 (21.0/1.3)
|   |   F10 > 1.400000 :
|   |   |   F1 <= 1.240000 :
|   |   |   |   F1 <= 0.350000 : 0 (7.0/1.3)
|   |   |   |   F1 > 0.350000 :
|   |   |   |   |   F10 <= 1.900000 :
|   |   |   |   |   |   F1 <= 0.870000 : 1 (2.0/1.0)
|   |   |   |   |   |   F1 > 0.870000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 1.900000 : 1 (7.0/1.3)
|   |   |   F1 > 1.240000 : 0 (9.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 1.870000 : 1 (27.0/1.4)
|   |   F1 > 1.870000 :
|   |   |   F10 <= 3.700000 : 0 (4.0/2.2)
|   |   |   F10 > 3.700000 : 1 (3.0/1.1)
