YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 50% of 132 cases. 

MISCLASSIFICATION on training: 1.51515% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	65	1	0
0	1	65	1

MISCLASSIFICATION on test file: 19.3206% of 471 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	202	78	0
0	13	178	1

DECISION TREE (size=9)

F1 <= 0.130000 : 0 (43.0/2.6)
F1 > 0.130000 :
|   F10 <= 2.800000 :
|   |   F10 <= 2.000000 : 0 (15.0/1.3)
|   |   F10 > 2.000000 :
|   |   |   F1 <= 1.300000 : 1 (12.0/2.5)
|   |   |   F1 > 1.300000 : 0 (8.0/1.3)
|   F10 > 2.800000 : 1 (54.0/1.4)
