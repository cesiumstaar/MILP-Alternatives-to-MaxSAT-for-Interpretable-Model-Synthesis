YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.7647% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	42	1	0
0	0	25	1

MISCLASSIFICATION on test file: 9.63455% of 301 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	184	25	0
0	4	88	1

DECISION TREE (size=15)

F1 <= 0.140000 : 0 (21.0/1.3)
F1 > 0.140000 :
|   F10 <= 2.000000 :
|   |   F10 <= 1.500000 : 0 (12.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 0.840000 :
|   |   |   |   F1 <= 0.400000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.400000 : 1 (4.0/1.2)
|   |   |   F1 > 0.840000 : 0 (5.0/1.2)
|   F10 > 2.000000 :
|   |   F10 <= 2.700000 :
|   |   |   F1 <= 1.440000 : 1 (8.0/2.4)
|   |   |   F1 > 1.440000 : 0 (2.0/1.0)
|   |   F10 > 2.700000 : 1 (14.0/1.3)
