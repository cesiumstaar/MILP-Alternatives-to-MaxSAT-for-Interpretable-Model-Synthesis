YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	54	1	0
0	0	45	1

MISCLASSIFICATION on test file: 18.9723% of 759 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	380	128	0
0	16	235	1

DECISION TREE (size=13)

F1 <= -0.220000 : 0 (38.0/1.4)
F1 > -0.220000 :
|   F10 <= 2.700000 :
|   |   F1 <= 1.210000 :
|   |   |   F10 <= 1.300000 : 0 (2.0/1.0)
|   |   |   F10 > 1.300000 : 1 (11.0/2.5)
|   |   F1 > 1.210000 : 0 (11.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 2.210000 : 1 (33.0/1.4)
|   |   F1 > 2.210000 :
|   |   |   F10 <= 4.400000 : 0 (3.0/1.1)
|   |   |   F10 > 4.400000 : 1 (2.0/1.0)
