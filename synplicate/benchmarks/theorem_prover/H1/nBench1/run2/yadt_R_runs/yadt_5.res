YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 49% of 100 cases. 

MISCLASSIFICATION on training: 4% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	50	1	0
0	3	46	1

MISCLASSIFICATION on test file: 19.8953% of 382 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	180	60	0
0	16	126	1

DECISION TREE (size=11)

F1 <= 0.400000 :
|   F10 <= 4.900000 : 0 (37.0/3.8)
|   F10 > 4.900000 : 1 (2.0/1.0)
F1 > 0.400000 :
|   F10 <= 2.800000 :
|   |   F1 <= 0.980000 :
|   |   |   F10 <= 1.500000 : 0 (3.0/1.1)
|   |   |   F10 > 1.500000 : 1 (9.0/1.3)
|   |   F1 > 0.980000 : 0 (13.0/2.5)
|   F10 > 2.800000 : 1 (36.0/2.6)
