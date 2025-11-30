YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.9048% of 84 cases. 

MISCLASSIFICATION on training: 2.38095% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	52	1	0
0	1	30	1

MISCLASSIFICATION on test file: 14.7929% of 338 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	208	29	0
0	21	80	1

DECISION TREE (size=13)

F1 <= 0.140000 : 0 (27.0/1.4)
F1 > 0.140000 :
|   F10 <= 2.000000 :
|   |   F10 <= 1.500000 : 0 (14.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 0.840000 :
|   |   |   |   F1 <= 0.400000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.400000 : 1 (4.0/1.2)
|   |   |   F1 > 0.840000 : 0 (6.0/1.2)
|   F10 > 2.000000 :
|   |   F1 <= 1.520000 : 1 (27.0/2.6)
|   |   F1 > 1.520000 : 0 (4.0/2.2)
