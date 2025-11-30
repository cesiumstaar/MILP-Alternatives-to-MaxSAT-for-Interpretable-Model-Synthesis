YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 35% of 20 cases. 

MISCLASSIFICATION on training: 0% of 20 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	7	0	0
0	0	13	1

MISCLASSIFICATION on test file: 37.2222% of 180 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	46	62	0
0	5	67	1

DECISION TREE (size=7)

F10 <= 2.300000 :
|   F1 <= 0.590000 : 0 (5.0/1.2)
|   F1 > 0.590000 :
|   |   F10 <= 1.600000 : 0 (2.0/1.0)
|   |   F10 > 1.600000 : 1 (3.0/1.1)
F10 > 2.300000 : 1 (10.0/1.3)
