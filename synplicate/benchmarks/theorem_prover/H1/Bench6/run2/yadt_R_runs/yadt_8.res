YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.1892% of 148 cases. 

MISCLASSIFICATION on training: 2.02703% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	89	1	0
0	2	56	1

MISCLASSIFICATION on test file: 12.7312% of 919 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	541	93	0
0	24	261	1

DECISION TREE (size=17)

F1 <= -0.260000 : 0 (53.0/2.6)
F1 > -0.260000 :
|   F10 <= 1.600000 : 0 (20.0/1.3)
|   F10 > 1.600000 :
|   |   F1 <= 1.630000 :
|   |   |   F1 <= 0.000000 :
|   |   |   |   F10 <= 3.900000 : 0 (2.0/1.0)
|   |   |   |   F10 > 3.900000 : 1 (3.0/1.1)
|   |   |   F1 > 0.000000 : 1 (42.0/2.6)
|   |   F1 > 1.630000 :
|   |   |   F10 <= 3.600000 : 0 (14.0/2.5)
|   |   |   F10 > 3.600000 :
|   |   |   |   F1 <= 2.630000 : 1 (9.0/1.3)
|   |   |   |   F1 > 2.630000 :
|   |   |   |   |   F10 <= 4.600000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 4.600000 : 1 (3.0/1.1)
