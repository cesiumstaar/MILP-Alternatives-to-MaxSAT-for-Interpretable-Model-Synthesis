YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.4444% of 36 cases. 

MISCLASSIFICATION on training: 2.77778% of 36 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	19	1	0
0	0	16	1

MISCLASSIFICATION on test file: 17.9724% of 217 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	115	32	0
0	7	63	1

DECISION TREE (size=9)

F10 <= 2.500000 :
|   F10 <= 1.300000 : 0 (9.0/1.3)
|   F10 > 1.300000 :
|   |   F1 <= 0.440000 : 0 (6.0/1.2)
|   |   F1 > 0.440000 :
|   |   |   F1 <= 0.800000 : 1 (5.0/1.2)
|   |   |   F1 > 0.800000 : 0 (4.0/1.2)
F10 > 2.500000 : 1 (12.0/2.5)
