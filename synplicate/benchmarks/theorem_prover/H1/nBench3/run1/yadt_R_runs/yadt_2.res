YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 40.3846% of 52 cases. 

MISCLASSIFICATION on training: 3.84615% of 52 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	31	0	0
0	2	19	1

MISCLASSIFICATION on test file: 9.65251% of 259 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	158	19	0
0	6	76	1

DECISION TREE (size=11)

F1 <= 0.140000 : 0 (13.0/1.3)
F1 > 0.140000 :
|   F10 <= 1.700000 : 0 (15.0/3.7)
|   F10 > 1.700000 :
|   |   F10 <= 2.700000 :
|   |   |   F1 <= 1.190000 :
|   |   |   |   F1 <= 0.340000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.340000 : 1 (8.0/1.3)
|   |   |   F1 > 1.190000 : 0 (3.0/1.1)
|   |   F10 > 2.700000 : 1 (11.0/1.3)
