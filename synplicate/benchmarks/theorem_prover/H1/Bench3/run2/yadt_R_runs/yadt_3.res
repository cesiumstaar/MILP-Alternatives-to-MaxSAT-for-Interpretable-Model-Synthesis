YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.7059% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	26	1	0
0	0	41	1

MISCLASSIFICATION on test file: 18.7097% of 310 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	122	52	0
0	6	130	1

DECISION TREE (size=9)

F1 <= 0.100000 : 0 (19.0/1.3)
F1 > 0.100000 :
|   F10 <= 2.400000 :
|   |   F10 <= 1.600000 : 0 (5.0/1.2)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 0.950000 : 1 (5.0/2.3)
|   |   |   F1 > 0.950000 : 0 (2.0/1.0)
|   F10 > 2.400000 : 1 (37.0/1.4)
