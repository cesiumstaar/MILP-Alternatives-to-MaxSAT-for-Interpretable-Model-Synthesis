YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.8049% of 164 cases. 

MISCLASSIFICATION on training: 1.21951% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	102	0	0
0	2	60	1

MISCLASSIFICATION on test file: 16.5109% of 963 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	533	137	0
0	22	271	1

DECISION TREE (size=15)

F1 <= -0.430000 : 0 (60.0/1.4)
F1 > -0.430000 :
|   F10 <= 2.600000 :
|   |   F1 <= 1.240000 :
|   |   |   F10 <= 1.600000 : 0 (12.0/2.5)
|   |   |   F10 > 1.600000 :
|   |   |   |   F1 <= 0.290000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.290000 : 1 (7.0/1.3)
|   |   F1 > 1.240000 : 0 (22.0/1.3)
|   F10 > 2.600000 :
|   |   F1 <= 2.240000 : 1 (48.0/1.4)
|   |   F1 > 2.240000 :
|   |   |   F10 <= 4.000000 : 0 (6.0/2.3)
|   |   |   F10 > 4.000000 : 1 (5.0/1.2)
