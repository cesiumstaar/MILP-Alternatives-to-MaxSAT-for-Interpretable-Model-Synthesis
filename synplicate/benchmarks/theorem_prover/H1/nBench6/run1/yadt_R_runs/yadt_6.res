YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 40.5172% of 116 cases. 

MISCLASSIFICATION on training: 1.72414% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	68	1	0
0	1	46	1

MISCLASSIFICATION on test file: 13.1026% of 809 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	473	89	0
0	17	230	1

DECISION TREE (size=15)

F1 <= 0.110000 : 0 (42.0/2.6)
F1 > 0.110000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (15.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 1.190000 :
|   |   |   |   F1 <= 0.450000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.450000 : 1 (3.0/1.1)
|   |   |   F1 > 1.190000 : 0 (5.0/1.2)
|   F10 > 2.500000 :
|   |   F1 <= 1.910000 : 1 (35.0/1.4)
|   |   F1 > 1.910000 :
|   |   |   F10 <= 3.400000 : 0 (4.0/1.2)
|   |   |   F10 > 3.400000 : 1 (9.0/2.4)
