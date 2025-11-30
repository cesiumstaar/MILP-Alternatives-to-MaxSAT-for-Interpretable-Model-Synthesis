YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.4615% of 52 cases. 

MISCLASSIFICATION on training: 3.84615% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	31	1	0
0	1	19	1

MISCLASSIFICATION on test file: 12.4807% of 649 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	399	62	0
0	19	169	1

DECISION TREE (size=9)

F1 <= -0.760000 : 0 (18.0/1.3)
F1 > -0.760000 :
|   F10 <= 2.600000 :
|   |   F1 <= 0.350000 : 0 (7.0/1.3)
|   |   F1 > 0.350000 :
|   |   |   F1 <= 0.870000 : 1 (5.0/1.2)
|   |   |   F1 > 0.870000 : 0 (7.0/2.4)
|   F10 > 2.600000 : 1 (15.0/2.5)
