YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.5517% of 116 cases. 

MISCLASSIFICATION on training: 2.58621% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	61	1	0
0	2	52	1

MISCLASSIFICATION on test file: 18.6158% of 419 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	200	69	0
0	9	141	1

DECISION TREE (size=17)

F1 <= 0.400000 :
|   F1 <= -0.300000 : 0 (24.0/1.3)
|   F1 > -0.300000 :
|   |   F10 <= 4.900000 :
|   |   |   F1 <= -0.020000 : 0 (10.0/1.3)
|   |   |   F1 > -0.020000 :
|   |   |   |   F10 <= 3.200000 : 0 (9.0/2.4)
|   |   |   |   F10 > 3.200000 : 1 (2.0/1.0)
|   |   F10 > 4.900000 : 1 (2.0/1.0)
F1 > 0.400000 :
|   F10 <= 2.800000 :
|   |   F1 <= 0.980000 :
|   |   |   F10 <= 1.500000 : 0 (4.0/1.2)
|   |   |   F10 > 1.500000 : 1 (9.0/1.3)
|   |   F1 > 0.980000 : 0 (16.0/2.5)
|   F10 > 2.800000 : 1 (40.0/2.6)
