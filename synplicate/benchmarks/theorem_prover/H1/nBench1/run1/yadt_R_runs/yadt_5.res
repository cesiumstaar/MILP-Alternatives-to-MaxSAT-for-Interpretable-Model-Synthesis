YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	56	0	0
0	1	43	1

MISCLASSIFICATION on test file: 16.7539% of 382 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	211	50	0
0	14	107	1

DECISION TREE (size=11)

F1 <= 0.180000 : 0 (35.0/2.6)
F1 > 0.180000 :
|   F10 <= 2.500000 :
|   |   F1 <= 0.800000 :
|   |   |   F10 <= 1.300000 : 0 (5.0/1.2)
|   |   |   F10 > 1.300000 :
|   |   |   |   F1 <= 0.440000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.440000 : 1 (8.0/1.3)
|   |   F1 > 0.800000 : 0 (14.0/1.3)
|   F10 > 2.500000 : 1 (35.0/1.4)
