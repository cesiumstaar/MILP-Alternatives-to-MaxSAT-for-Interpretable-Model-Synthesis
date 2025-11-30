YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 25% of 20 cases. 

MISCLASSIFICATION on training: 0% of 20 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	5	0	0
0	0	15	1

MISCLASSIFICATION on test file: 48.8889% of 180 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	19	87	0
0	1	73	1

DECISION TREE (size=7)

F10 <= 2.800000 :
|   F1 <= 0.310000 : 0 (3.0/1.1)
|   F1 > 0.310000 :
|   |   F10 <= 1.200000 : 0 (2.0/1.0)
|   |   F10 > 1.200000 : 1 (6.0/1.2)
F10 > 2.800000 : 1 (9.0/1.3)
