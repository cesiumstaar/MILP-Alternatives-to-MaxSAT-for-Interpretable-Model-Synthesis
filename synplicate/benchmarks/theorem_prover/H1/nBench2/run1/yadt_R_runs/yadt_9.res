YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.5854% of 164 cases. 

MISCLASSIFICATION on training: 1.82927% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	103	1	0
0	2	58	1

MISCLASSIFICATION on test file: 14.8496% of 1064 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	611	131	0
0	27	295	1

DECISION TREE (size=9)

F1 <= -0.680000 : 0 (60.0/1.4)
F1 > -0.680000 :
|   F10 <= 2.500000 : 0 (40.0/3.8)
|   F10 > 2.500000 :
|   |   F10 <= 3.300000 :
|   |   |   F1 <= 1.320000 : 1 (11.0/2.5)
|   |   |   F1 > 1.320000 : 0 (5.0/1.2)
|   |   F10 > 3.300000 : 1 (48.0/1.4)
