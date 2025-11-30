YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 35.3659% of 164 cases. 

MISCLASSIFICATION on training: 1.21951% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	105	1	0
0	1	57	1

MISCLASSIFICATION on test file: 13.5841% of 957 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	567	110	0
0	20	260	1

DECISION TREE (size=15)

F1 <= 0.110000 : 0 (62.0/2.6)
F1 > 0.110000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (26.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 1.190000 :
|   |   |   |   F1 <= 0.300000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.300000 : 1 (5.0/1.2)
|   |   |   F1 > 1.190000 : 0 (9.0/1.3)
|   F10 > 2.500000 :
|   |   F1 <= 1.910000 : 1 (42.0/1.4)
|   |   F1 > 1.910000 :
|   |   |   F10 <= 3.500000 : 0 (6.0/1.2)
|   |   |   F10 > 3.500000 : 1 (11.0/2.5)
