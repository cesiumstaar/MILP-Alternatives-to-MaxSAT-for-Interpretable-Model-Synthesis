YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.6364% of 132 cases. 

MISCLASSIFICATION on training: 1.51515% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	79	2	0
0	0	51	1

MISCLASSIFICATION on test file: 13.9959% of 493 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	258	67	0
0	2	166	1

DECISION TREE (size=11)

F1 <= -0.030000 : 0 (42.0/1.4)
F1 > -0.030000 :
|   F10 <= 2.800000 :
|   |   F10 <= 1.600000 : 0 (24.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.380000 :
|   |   |   |   F1 <= 0.290000 : 0 (5.0/1.2)
|   |   |   |   F1 > 0.290000 : 1 (15.0/2.5)
|   |   |   F1 > 1.380000 : 0 (8.0/1.3)
|   F10 > 2.800000 : 1 (38.0/2.6)
