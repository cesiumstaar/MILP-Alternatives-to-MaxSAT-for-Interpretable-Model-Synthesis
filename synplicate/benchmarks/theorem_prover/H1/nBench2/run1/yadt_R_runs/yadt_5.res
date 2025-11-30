YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38% of 100 cases. 

MISCLASSIFICATION on training: 2% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	62	0	0
0	2	36	1

MISCLASSIFICATION on test file: 20.5282% of 833 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	425	138	0
0	33	237	1

DECISION TREE (size=9)

F1 <= -0.690000 : 0 (38.0/1.4)
F1 > -0.690000 :
|   F10 <= 2.700000 : 0 (24.0/3.7)
|   F10 > 2.700000 :
|   |   F10 <= 3.300000 :
|   |   |   F1 <= 1.210000 : 1 (5.0/1.2)
|   |   |   F1 > 1.210000 : 0 (2.0/1.0)
|   |   F10 > 3.300000 : 1 (31.0/1.4)
