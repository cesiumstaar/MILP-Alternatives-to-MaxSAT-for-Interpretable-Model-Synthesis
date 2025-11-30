YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.4444% of 36 cases. 

MISCLASSIFICATION on training: 2.77778% of 36 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	19	1	0
0	0	16	1

MISCLASSIFICATION on test file: 21.659% of 217 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	99	45	0
0	2	71	1

DECISION TREE (size=11)

F10 <= 2.000000 :
|   F10 <= 1.300000 : 0 (9.0/1.3)
|   F10 > 1.300000 :
|   |   F1 <= 0.400000 : 0 (5.0/1.2)
|   |   F1 > 0.400000 :
|   |   |   F1 <= 0.840000 : 1 (4.0/1.2)
|   |   |   F1 > 0.840000 : 0 (3.0/1.1)
F10 > 2.000000 :
|   F1 <= -0.250000 : 0 (2.0/1.0)
|   F1 > -0.250000 : 1 (13.0/2.5)
