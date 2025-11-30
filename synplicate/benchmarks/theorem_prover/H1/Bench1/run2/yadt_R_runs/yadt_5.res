YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 47% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	46	1	0
0	0	53	1

MISCLASSIFICATION on test file: 18.136% of 397 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	168	64	0
0	8	157	1

DECISION TREE (size=9)

F1 <= 0.130000 : 0 (31.0/1.4)
F1 > 0.130000 :
|   F10 <= 2.700000 :
|   |   F10 <= 2.000000 : 0 (11.0/1.3)
|   |   F10 > 2.000000 :
|   |   |   F1 <= 1.560000 : 1 (10.0/2.4)
|   |   |   F1 > 1.560000 : 0 (4.0/1.2)
|   F10 > 2.700000 : 1 (44.0/1.4)
