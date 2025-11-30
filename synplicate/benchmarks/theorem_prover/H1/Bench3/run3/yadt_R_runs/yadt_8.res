YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.4865% of 148 cases. 

MISCLASSIFICATION on training: 2.7027% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	92	2	0
0	2	52	1

MISCLASSIFICATION on test file: 16.2617% of 535 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	293	72	0
0	15	155	1

DECISION TREE (size=11)

F1 <= 0.170000 : 0 (54.0/3.8)
F1 > 0.170000 :
|   F10 <= 2.700000 :
|   |   F1 <= 1.000000 :
|   |   |   F10 <= 1.300000 : 0 (11.0/1.3)
|   |   |   F10 > 1.300000 :
|   |   |   |   F1 <= 0.310000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.310000 : 1 (11.0/2.5)
|   |   F1 > 1.000000 : 0 (26.0/1.3)
|   F10 > 2.700000 : 1 (43.0/2.6)
