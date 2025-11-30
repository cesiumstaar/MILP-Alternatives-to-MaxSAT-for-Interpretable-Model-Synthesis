YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 40.3846% of 52 cases. 

MISCLASSIFICATION on training: 3.84615% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	30	1	0
0	1	20	1

MISCLASSIFICATION on test file: 14.0152% of 264 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	147	28	0
0	9	80	1

DECISION TREE (size=9)

F10 <= 1.900000 : 0 (16.0/1.3)
F10 > 1.900000 :
|   F1 <= 0.210000 : 0 (12.0/2.5)
|   F1 > 0.210000 :
|   |   F10 <= 2.700000 :
|   |   |   F1 <= 1.130000 : 1 (7.0/2.4)
|   |   |   F1 > 1.130000 : 0 (3.0/1.1)
|   |   F10 > 2.700000 : 1 (14.0/1.3)
