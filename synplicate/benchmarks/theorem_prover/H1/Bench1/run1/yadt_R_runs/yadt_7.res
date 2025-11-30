YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.2121% of 132 cases. 

MISCLASSIFICATION on training: 1.51515% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	69	2	0
0	0	61	1

MISCLASSIFICATION on test file: 18.4946% of 465 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	204	84	0
0	2	175	1

DECISION TREE (size=13)

F1 <= 0.100000 : 0 (43.0/1.4)
F1 > 0.100000 :
|   F10 <= 2.600000 :
|   |   F1 <= 1.320000 :
|   |   |   F10 <= 1.400000 : 0 (8.0/1.3)
|   |   |   F10 > 1.400000 :
|   |   |   |   F10 <= 2.200000 :
|   |   |   |   |   F1 <= 0.330000 : 0 (3.0/1.1)
|   |   |   |   |   F1 > 0.330000 : 1 (6.0/2.3)
|   |   |   |   F10 > 2.200000 : 1 (6.0/1.2)
|   |   F1 > 1.320000 : 0 (15.0/1.3)
|   F10 > 2.600000 : 1 (51.0/2.6)
