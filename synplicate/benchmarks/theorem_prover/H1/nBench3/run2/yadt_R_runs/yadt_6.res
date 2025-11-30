YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.7931% of 116 cases. 

MISCLASSIFICATION on training: 1.72414% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	69	2	0
0	0	45	1

MISCLASSIFICATION on test file: 18.2033% of 423 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	236	45	0
0	32	110	1

DECISION TREE (size=11)

F1 <= 0.120000 : 0 (39.0/1.4)
F1 > 0.120000 :
|   F10 <= 1.600000 : 0 (22.0/1.3)
|   F10 > 1.600000 :
|   |   F1 <= 1.630000 :
|   |   |   F10 <= 2.500000 :
|   |   |   |   F1 <= 1.220000 : 1 (12.0/3.6)
|   |   |   |   F1 > 1.220000 : 0 (2.0/1.0)
|   |   |   F10 > 2.500000 : 1 (35.0/1.4)
|   |   F1 > 1.630000 : 0 (6.0/1.2)
