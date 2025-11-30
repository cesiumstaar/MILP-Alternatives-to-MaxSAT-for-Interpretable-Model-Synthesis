YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	60	1	0
0	0	39	1

MISCLASSIFICATION on test file: 16.273% of 381 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	216	36	0
0	26	103	1

DECISION TREE (size=11)

F10 <= 1.600000 : 0 (31.0/1.4)
F10 > 1.600000 :
|   F1 <= 0.120000 : 0 (21.0/1.3)
|   F1 > 0.120000 :
|   |   F1 <= 1.630000 :
|   |   |   F10 <= 2.500000 :
|   |   |   |   F1 <= 1.180000 : 1 (11.0/2.5)
|   |   |   |   F1 > 1.180000 : 0 (2.0/1.0)
|   |   |   F10 > 2.500000 : 1 (29.0/1.4)
|   |   F1 > 1.630000 : 0 (6.0/1.2)
