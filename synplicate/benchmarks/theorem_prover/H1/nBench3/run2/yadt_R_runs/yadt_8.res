YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.1622% of 148 cases. 

MISCLASSIFICATION on training: 1.35135% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	91	2	0
0	0	55	1

MISCLASSIFICATION on test file: 16.4706% of 510 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	268	77	0
0	7	158	1

DECISION TREE (size=17)

F10 <= 1.600000 : 0 (45.0/1.4)
F10 > 1.600000 :
|   F1 <= 0.120000 :
|   |   F10 <= 4.500000 : 0 (31.0/1.4)
|   |   F10 > 4.500000 :
|   |   |   F1 <= -0.370000 : 0 (4.0/1.2)
|   |   |   F1 > -0.370000 : 1 (2.0/1.0)
|   F1 > 0.120000 :
|   |   F1 <= 1.630000 :
|   |   |   F10 <= 2.500000 :
|   |   |   |   F1 <= 1.220000 : 1 (12.0/3.6)
|   |   |   |   F1 > 1.220000 : 0 (3.0/1.1)
|   |   |   F10 > 2.500000 : 1 (41.0/1.4)
|   |   F1 > 1.630000 :
|   |   |   F10 <= 3.100000 : 0 (8.0/1.3)
|   |   |   F10 > 3.100000 : 1 (2.0/1.0)
