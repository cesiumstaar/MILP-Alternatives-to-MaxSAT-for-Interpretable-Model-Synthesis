YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 40.4908% of 163 cases. 

MISCLASSIFICATION on training: 1.84049% of 163 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	96	1	0
0	2	64	1

MISCLASSIFICATION on test file: 20.1465% of 546 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	265	100	0
0	10	171	1

DECISION TREE (size=13)

F1 <= 0.290000 :
|   F10 <= 5.000000 : 0 (54.0/1.4)
|   F10 > 5.000000 :
|   |   F1 <= -0.360000 : 0 (2.0/1.0)
|   |   F1 > -0.360000 : 1 (2.0/1.0)
F1 > 0.290000 :
|   F10 <= 2.600000 :
|   |   F10 <= 1.800000 : 0 (30.0/2.6)
|   |   F10 > 1.800000 :
|   |   |   F1 <= 0.910000 : 1 (11.0/2.5)
|   |   |   F1 > 0.910000 : 0 (12.0/2.5)
|   F10 > 2.600000 : 1 (52.0/1.4)
