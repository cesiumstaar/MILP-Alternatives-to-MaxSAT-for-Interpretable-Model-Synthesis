YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.3939% of 132 cases. 

MISCLASSIFICATION on training: 0.757576% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	80	0	0
0	1	51	1

MISCLASSIFICATION on test file: 11.9767% of 860 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	513	73	0
0	30	244	1

DECISION TREE (size=17)

F1 <= -0.260000 : 0 (45.0/2.6)
F1 > -0.260000 :
|   F10 <= 1.600000 : 0 (20.0/1.3)
|   F10 > 1.600000 :
|   |   F1 <= 1.340000 :
|   |   |   F1 <= 0.000000 :
|   |   |   |   F10 <= 3.900000 : 0 (2.0/1.0)
|   |   |   |   F10 > 3.900000 : 1 (3.0/1.1)
|   |   |   F1 > 0.000000 : 1 (37.0/1.4)
|   |   F1 > 1.340000 :
|   |   |   F10 <= 4.400000 :
|   |   |   |   F1 <= 2.000000 :
|   |   |   |   |   F10 <= 3.000000 : 0 (4.0/1.2)
|   |   |   |   |   F10 > 3.000000 : 1 (4.0/1.2)
|   |   |   |   F1 > 2.000000 : 0 (10.0/1.3)
|   |   |   F10 > 4.400000 : 1 (7.0/1.3)
