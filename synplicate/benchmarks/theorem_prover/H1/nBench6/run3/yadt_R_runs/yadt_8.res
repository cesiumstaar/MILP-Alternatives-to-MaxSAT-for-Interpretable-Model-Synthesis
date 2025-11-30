YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.415% of 147 cases. 

MISCLASSIFICATION on training: 0.680272% of 147 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	91	1	0
0	0	55	1

MISCLASSIFICATION on test file: 11.7778% of 900 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	548	92	0
0	14	246	1

DECISION TREE (size=19)

F1 <= -0.280000 : 0 (53.0/1.4)
F1 > -0.280000 :
|   F10 <= 2.600000 :
|   |   F1 <= 1.190000 :
|   |   |   F10 <= 1.200000 : 0 (8.0/1.3)
|   |   |   F10 > 1.200000 :
|   |   |   |   F1 <= 0.280000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.280000 : 1 (10.0/1.3)
|   |   F1 > 1.190000 : 0 (21.0/1.3)
|   F10 > 2.600000 :
|   |   F1 <= 1.890000 :
|   |   |   F1 <= -0.110000 :
|   |   |   |   F10 <= 4.600000 : 0 (2.0/1.0)
|   |   |   |   F10 > 4.600000 : 1 (2.0/1.0)
|   |   |   F1 > -0.110000 : 1 (36.0/1.4)
|   |   F1 > 1.890000 :
|   |   |   F10 <= 3.700000 : 0 (5.0/1.2)
|   |   |   F10 > 3.700000 : 1 (8.0/2.4)
