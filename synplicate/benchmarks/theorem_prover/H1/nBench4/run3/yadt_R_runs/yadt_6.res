YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.2414% of 116 cases. 

MISCLASSIFICATION on training: 0% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	67	0	0
0	0	49	1

MISCLASSIFICATION on test file: 15.7263% of 833 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	448	116	0
0	15	254	1

DECISION TREE (size=15)

F1 <= -0.550000 : 0 (40.0/1.4)
F1 > -0.550000 :
|   F10 <= 2.600000 :
|   |   F10 <= 1.600000 : 0 (15.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.240000 :
|   |   |   |   F1 <= 0.290000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.290000 : 1 (6.0/1.2)
|   |   |   F1 > 1.240000 : 0 (6.0/1.2)
|   F10 > 2.600000 :
|   |   F1 <= 2.240000 : 1 (39.0/1.4)
|   |   F1 > 2.240000 :
|   |   |   F10 <= 3.400000 : 0 (2.0/1.0)
|   |   |   F10 > 3.400000 : 1 (4.0/1.2)
