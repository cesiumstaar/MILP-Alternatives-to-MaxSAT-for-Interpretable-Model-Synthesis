YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 48.8095% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	43	0	0
0	1	40	1

MISCLASSIFICATION on test file: 15.3846% of 338 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	170	45	0
0	7	116	1

DECISION TREE (size=11)

F1 <= 0.170000 : 0 (26.0/2.6)
F1 > 0.170000 :
|   F10 <= 2.800000 :
|   |   F10 <= 1.600000 : 0 (11.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.280000 :
|   |   |   |   F1 <= 0.280000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.280000 : 1 (9.0/1.3)
|   |   |   F1 > 1.280000 : 0 (5.0/1.2)
|   F10 > 2.800000 : 1 (31.0/1.4)
