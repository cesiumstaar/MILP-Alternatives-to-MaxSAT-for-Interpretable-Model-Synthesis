YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45% of 20 cases. 

MISCLASSIFICATION on training: 0% of 20 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	9	0	0
0	0	11	1

MISCLASSIFICATION on test file: 44.8% of 500 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	132	215	0
0	9	144	1

DECISION TREE (size=7)

F10 <= 2.900000 :
|   F10 <= 1.900000 : 0 (7.0/1.3)
|   F10 > 1.900000 :
|   |   F1 <= 0.480000 : 0 (2.0/1.0)
|   |   F1 > 0.480000 : 1 (2.0/1.0)
F10 > 2.900000 : 1 (9.0/1.3)
