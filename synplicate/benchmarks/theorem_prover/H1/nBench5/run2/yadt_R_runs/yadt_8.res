YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.5946% of 148 cases. 

MISCLASSIFICATION on training: 1.35135% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	66	0	0
0	2	80	1

MISCLASSIFICATION on test file: 20.1258% of 477 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	189	89	0
0	7	192	1

DECISION TREE (size=17)

F1 <= -0.190000 : 0 (27.0/1.4)
F1 > -0.190000 :
|   F10 <= 3.000000 :
|   |   F10 <= 1.600000 : 0 (17.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.620000 :
|   |   |   |   F1 <= 0.310000 : 0 (5.0/1.2)
|   |   |   |   F1 > 0.310000 :
|   |   |   |   |   F1 <= 1.430000 : 1 (13.0/1.3)
|   |   |   |   |   F1 > 1.430000 : 0 (3.0/2.1)
|   |   |   F1 > 1.620000 : 0 (7.0/1.3)
|   F10 > 3.000000 :
|   |   F1 <= 0.000000 :
|   |   |   F10 <= 4.800000 : 0 (9.0/2.4)
|   |   |   F10 > 4.800000 : 1 (8.0/1.3)
|   |   F1 > 0.000000 : 1 (59.0/1.4)
