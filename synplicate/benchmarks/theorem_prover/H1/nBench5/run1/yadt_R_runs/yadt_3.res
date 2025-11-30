YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 48.5294% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	33	0	0
0	1	34	1

MISCLASSIFICATION on test file: 16.8919% of 296 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	142	43	0
0	7	104	1

DECISION TREE (size=11)

F1 <= 0.170000 : 0 (20.0/2.5)
F1 > 0.170000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.600000 : 0 (9.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.280000 :
|   |   |   |   F1 <= 0.280000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.280000 : 1 (6.0/1.2)
|   |   |   F1 > 1.280000 : 0 (3.0/1.1)
|   F10 > 2.500000 : 1 (28.0/1.4)
