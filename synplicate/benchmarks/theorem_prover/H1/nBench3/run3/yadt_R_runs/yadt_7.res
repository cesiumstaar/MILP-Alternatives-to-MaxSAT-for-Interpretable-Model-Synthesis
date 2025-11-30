YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.9697% of 132 cases. 

MISCLASSIFICATION on training: 2.27273% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	68	2	0
0	1	61	1

MISCLASSIFICATION on test file: 18.3932% of 473 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	225	76	0
0	11	161	1

DECISION TREE (size=13)

F1 <= 0.110000 : 0 (45.0/2.6)
F1 > 0.110000 :
|   F10 <= 2.900000 :
|   |   F1 <= 1.540000 :
|   |   |   F10 <= 1.900000 :
|   |   |   |   F10 <= 1.300000 : 0 (7.0/1.3)
|   |   |   |   F10 > 1.300000 :
|   |   |   |   |   F1 <= 0.780000 : 1 (2.0/1.0)
|   |   |   |   |   F1 > 0.780000 : 0 (3.0/1.1)
|   |   |   F10 > 1.900000 : 1 (14.0/2.5)
|   |   F1 > 1.540000 : 0 (14.0/1.3)
|   F10 > 2.900000 : 1 (47.0/2.6)
