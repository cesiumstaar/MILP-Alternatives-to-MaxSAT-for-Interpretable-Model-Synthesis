YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.3939% of 132 cases. 

MISCLASSIFICATION on training: 0.757576% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	80	0	0
0	1	51	1

MISCLASSIFICATION on test file: 16.3128% of 895 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	484	128	0
0	18	265	1

DECISION TREE (size=15)

F1 <= -0.430000 : 0 (50.0/1.4)
F1 > -0.430000 :
|   F10 <= 2.600000 :
|   |   F1 <= 1.240000 :
|   |   |   F1 <= 0.290000 : 0 (7.0/1.3)
|   |   |   F1 > 0.290000 :
|   |   |   |   F10 <= 1.600000 : 0 (8.0/2.4)
|   |   |   |   F10 > 1.600000 : 1 (6.0/1.2)
|   |   F1 > 1.240000 : 0 (14.0/1.3)
|   F10 > 2.600000 :
|   |   F1 <= 2.240000 : 1 (41.0/1.4)
|   |   F1 > 2.240000 :
|   |   |   F10 <= 3.400000 : 0 (2.0/1.0)
|   |   |   F10 > 3.400000 : 1 (4.0/1.2)
