YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.0476% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	36	1	0
0	0	47	1

MISCLASSIFICATION on test file: 16.0458% of 349 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	149	49	0
0	7	144	1

DECISION TREE (size=9)

F1 <= 0.130000 : 0 (25.0/1.3)
F1 > 0.130000 :
|   F10 <= 2.700000 :
|   |   F10 <= 2.000000 : 0 (8.0/1.3)
|   |   F10 > 2.000000 :
|   |   |   F1 <= 1.560000 : 1 (9.0/2.4)
|   |   |   F1 > 1.560000 : 0 (3.0/1.1)
|   F10 > 2.700000 : 1 (39.0/1.4)
