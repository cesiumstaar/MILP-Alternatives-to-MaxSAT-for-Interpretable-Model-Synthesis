YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 47.619% of 84 cases. 

MISCLASSIFICATION on training: 3.57143% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	39	1	0
0	2	42	1

MISCLASSIFICATION on test file: 23.2143% of 728 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	317	150	0
0	19	242	1

DECISION TREE (size=15)

F1 <= -0.140000 : 0 (25.0/2.5)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F1 <= 0.920000 :
|   |   |   F1 <= 0.400000 : 0 (2.0/1.0)
|   |   |   F1 > 0.400000 : 1 (8.0/2.4)
|   |   F1 > 0.920000 : 0 (9.0/1.3)
|   F10 > 2.700000 :
|   |   F10 <= 4.600000 :
|   |   |   F1 <= -0.120000 : 0 (2.0/1.0)
|   |   |   F1 > -0.120000 :
|   |   |   |   F1 <= 1.880000 : 1 (16.0/1.3)
|   |   |   |   F1 > 1.880000 : 0 (3.0/2.1)
|   |   F10 > 4.600000 : 1 (19.0/1.3)
