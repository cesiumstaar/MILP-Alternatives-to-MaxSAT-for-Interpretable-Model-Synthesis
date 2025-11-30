YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.2308% of 52 cases. 

MISCLASSIFICATION on training: 1.92308% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	22	1	0
0	0	29	1

MISCLASSIFICATION on test file: 20.4633% of 259 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	100	51	0
0	2	106	1

DECISION TREE (size=9)

F1 <= 0.240000 : 0 (15.0/1.3)
F1 > 0.240000 :
|   F10 <= 2.400000 :
|   |   F10 <= 1.400000 : 0 (4.0/1.2)
|   |   F10 > 1.400000 :
|   |   |   F1 <= 0.980000 : 1 (5.0/2.3)
|   |   |   F1 > 0.980000 : 0 (3.0/1.1)
|   F10 > 2.400000 : 1 (25.0/1.3)
