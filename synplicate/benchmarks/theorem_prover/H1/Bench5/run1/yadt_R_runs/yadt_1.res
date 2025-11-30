YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 33.3333% of 36 cases. 

MISCLASSIFICATION on training: 2.77778% of 36 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	11	1	0
0	0	24	1

MISCLASSIFICATION on test file: 30.1802% of 222 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	66	63	0
0	4	89	1

DECISION TREE (size=9)

F1 <= 0.310000 :
|   F1 <= -0.140000 : 0 (6.0/1.2)
|   F1 > -0.140000 :
|   |   F10 <= 2.800000 : 0 (3.0/1.1)
|   |   F10 > 2.800000 : 1 (2.0/1.0)
F1 > 0.310000 :
|   F10 <= 1.200000 : 0 (2.0/1.0)
|   F10 > 1.200000 : 1 (23.0/2.5)
