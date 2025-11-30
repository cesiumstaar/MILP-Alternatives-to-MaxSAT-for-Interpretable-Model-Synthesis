YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.1176% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	29	1	0
0	0	38	1

MISCLASSIFICATION on test file: 16.6113% of 301 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	130	44	0
0	6	121	1

DECISION TREE (size=9)

F1 <= 0.130000 : 0 (19.0/1.3)
F1 > 0.130000 :
|   F10 <= 2.000000 : 0 (8.0/1.3)
|   F10 > 2.000000 :
|   |   F10 <= 2.700000 :
|   |   |   F1 <= 1.560000 : 1 (8.0/2.4)
|   |   |   F1 > 1.560000 : 0 (2.0/1.0)
|   |   F10 > 2.700000 : 1 (31.0/1.4)
