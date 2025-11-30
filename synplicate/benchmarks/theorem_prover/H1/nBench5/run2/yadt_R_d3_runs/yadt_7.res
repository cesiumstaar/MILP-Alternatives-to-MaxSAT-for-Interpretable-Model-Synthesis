YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.697% of 132 cases. 

MISCLASSIFICATION on training: 6.06061% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	51	8	0
0	0	73	1

MISCLASSIFICATION on test file: 27.4157% of 445 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	140	115	0
0	7	183	1

DECISION TREE (size=11)

F1 <= 0.220000 :
|   F10 <= 4.800000 : 0 (37.0/1.4)
|   F10 > 4.800000 :
|   |   F1 <= -0.380000 : 0 (2.0/1.0)
|   |   F1 > -0.380000 : 1 (7.0/1.3)
F1 > 0.220000 :
|   F10 <= 3.000000 :
|   |   F10 <= 1.500000 : 0 (12.0/1.3)
|   |   F10 > 1.500000 : 1 (22.0/10.1)
|   F10 > 3.000000 : 1 (52.0/1.4)
