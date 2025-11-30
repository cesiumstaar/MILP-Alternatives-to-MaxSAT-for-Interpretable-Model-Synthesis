YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 49.4949% of 99 cases. 

MISCLASSIFICATION on training: 2.0202% of 99 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	48	2	0
0	0	49	1

MISCLASSIFICATION on test file: 18.0593% of 371 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	171	63	0
0	4	133	1

DECISION TREE (size=15)

F1 <= 0.290000 :
|   F10 <= 5.000000 : 0 (29.0/1.4)
|   F10 > 5.000000 : 1 (3.0/2.1)
F1 > 0.290000 :
|   F10 <= 2.600000 :
|   |   F10 <= 1.300000 : 0 (11.0/1.3)
|   |   F10 > 1.300000 :
|   |   |   F1 <= 1.470000 :
|   |   |   |   F10 <= 2.100000 :
|   |   |   |   |   F1 <= 0.720000 : 1 (4.0/2.2)
|   |   |   |   |   F1 > 0.720000 : 0 (3.0/1.1)
|   |   |   |   F10 > 2.100000 : 1 (5.0/1.2)
|   |   |   F1 > 1.470000 : 0 (5.0/1.2)
|   F10 > 2.600000 : 1 (39.0/1.4)
