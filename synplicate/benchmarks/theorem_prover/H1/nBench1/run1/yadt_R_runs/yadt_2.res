YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.3077% of 52 cases. 

MISCLASSIFICATION on training: 0% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	30	0	0
0	0	22	1

MISCLASSIFICATION on test file: 14.9606% of 254 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	145	29	0
0	9	71	1

DECISION TREE (size=11)

F10 <= 2.500000 :
|   F10 <= 1.300000 : 0 (14.0/1.3)
|   F10 > 1.300000 :
|   |   F1 <= 0.440000 : 0 (7.0/1.3)
|   |   F1 > 0.440000 :
|   |   |   F1 <= 0.800000 : 1 (6.0/1.2)
|   |   |   F1 > 0.800000 : 0 (5.0/1.2)
F10 > 2.500000 :
|   F1 <= -0.200000 : 0 (4.0/1.2)
|   F1 > -0.200000 : 1 (16.0/1.3)
