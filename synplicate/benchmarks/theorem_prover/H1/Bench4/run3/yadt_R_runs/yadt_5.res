YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 29% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	70	1	0
0	0	29	1

MISCLASSIFICATION on test file: 11.8665% of 809 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	549	58	0
0	38	164	1

DECISION TREE (size=11)

F10 <= 1.700000 : 0 (39.0/1.4)
F10 > 1.700000 :
|   F1 <= -0.140000 : 0 (22.0/1.3)
|   F1 > -0.140000 :
|   |   F1 <= 1.250000 : 1 (25.0/2.5)
|   |   F1 > 1.250000 :
|   |   |   F10 <= 2.800000 : 0 (7.0/1.3)
|   |   |   F10 > 2.800000 :
|   |   |   |   F1 <= 2.250000 : 1 (5.0/1.2)
|   |   |   |   F1 > 2.250000 : 0 (2.0/1.0)
