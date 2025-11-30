YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 33.7349% of 83 cases. 

MISCLASSIFICATION on training: 4.81928% of 83 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	54	1	0
0	3	25	1

MISCLASSIFICATION on test file: 12.0968% of 744 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	473	59	0
0	31	181	1

DECISION TREE (size=11)

F10 <= 2.000000 : 0 (37.0/3.8)
F10 > 2.000000 :
|   F1 <= -0.530000 : 0 (12.0/1.3)
|   F1 > -0.530000 :
|   |   F10 <= 3.500000 :
|   |   |   F1 <= 0.150000 : 0 (3.0/1.1)
|   |   |   F1 > 0.150000 :
|   |   |   |   F1 <= 1.470000 : 1 (8.0/1.3)
|   |   |   |   F1 > 1.470000 : 0 (5.0/2.3)
|   |   F10 > 3.500000 : 1 (18.0/2.5)
