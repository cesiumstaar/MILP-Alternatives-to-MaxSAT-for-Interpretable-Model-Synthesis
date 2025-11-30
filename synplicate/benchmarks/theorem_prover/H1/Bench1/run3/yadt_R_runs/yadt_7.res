YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.697% of 132 cases. 

MISCLASSIFICATION on training: 0.757576% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	58	1	0
0	0	73	1

MISCLASSIFICATION on test file: 17.7778% of 450 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	167	72	0
0	8	203	1

DECISION TREE (size=13)

F1 <= 0.000000 :
|   F1 <= -0.290000 : 0 (27.0/1.4)
|   F1 > -0.290000 :
|   |   F10 <= 4.300000 : 0 (10.0/1.3)
|   |   F10 > 4.300000 : 1 (5.0/2.3)
F1 > 0.000000 :
|   F10 <= 3.000000 :
|   |   F1 <= 1.330000 :
|   |   |   F10 <= 1.600000 : 0 (7.0/1.3)
|   |   |   F10 > 1.600000 : 1 (9.0/1.3)
|   |   F1 > 1.330000 : 0 (14.0/1.3)
|   F10 > 3.000000 : 1 (60.0/1.4)
