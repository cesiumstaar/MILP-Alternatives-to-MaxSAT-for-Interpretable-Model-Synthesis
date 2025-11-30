YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 40.404% of 99 cases. 

MISCLASSIFICATION on training: 1.0101% of 99 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	59	0	0
0	1	39	1

MISCLASSIFICATION on test file: 17.0487% of 698 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	391	101	0
0	18	188	1

DECISION TREE (size=17)

F1 <= -0.230000 :
|   F10 <= 5.500000 : 0 (34.0/1.4)
|   F10 > 5.500000 :
|   |   F1 <= -0.820000 : 0 (2.0/1.0)
|   |   F1 > -0.820000 : 1 (2.0/1.0)
F1 > -0.230000 :
|   F10 <= 2.300000 :
|   |   F1 <= 0.770000 :
|   |   |   F10 <= 2.100000 : 0 (7.0/2.4)
|   |   |   F10 > 2.100000 : 1 (2.0/1.0)
|   |   F1 > 0.770000 : 0 (12.0/1.3)
|   F10 > 2.300000 :
|   |   F1 <= 1.770000 : 1 (30.0/1.4)
|   |   F1 > 1.770000 :
|   |   |   F10 <= 4.200000 : 0 (5.0/1.2)
|   |   |   F10 > 4.200000 : 1 (5.0/1.2)
