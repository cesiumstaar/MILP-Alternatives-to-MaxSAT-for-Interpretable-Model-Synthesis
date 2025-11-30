YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.069% of 116 cases. 

MISCLASSIFICATION on training: 2.58621% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	72	1	0
0	2	41	1

MISCLASSIFICATION on test file: 13.0807% of 818 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	517	77	0
0	30	194	1

DECISION TREE (size=13)

F1 <= 0.160000 : 0 (42.0/2.6)
F1 > 0.160000 :
|   F10 <= 1.300000 : 0 (16.0/1.3)
|   F10 > 1.300000 :
|   |   F1 <= 1.630000 :
|   |   |   F10 <= 2.300000 :
|   |   |   |   F1 <= 0.970000 : 1 (8.0/2.4)
|   |   |   |   F1 > 0.970000 : 0 (3.0/1.1)
|   |   |   F10 > 2.300000 : 1 (29.0/1.4)
|   |   F1 > 1.630000 :
|   |   |   F10 <= 3.700000 : 0 (13.0/2.5)
|   |   |   F10 > 3.700000 : 1 (5.0/1.2)
