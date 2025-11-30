YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.3939% of 132 cases. 

MISCLASSIFICATION on training: 0% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	80	0	0
0	0	52	1

MISCLASSIFICATION on test file: 13.1635% of 471 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	270	54	0
0	8	139	1

DECISION TREE (size=15)

F1 <= -0.100000 : 0 (42.0/1.4)
F1 > -0.100000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.300000 : 0 (19.0/1.3)
|   |   F10 > 1.300000 :
|   |   |   F1 <= 1.160000 :
|   |   |   |   F1 <= 0.440000 : 0 (6.0/1.2)
|   |   |   |   F1 > 0.440000 :
|   |   |   |   |   F1 <= 0.800000 : 1 (8.0/1.3)
|   |   |   |   |   F1 > 0.800000 :
|   |   |   |   |   |   F10 <= 2.100000 : 0 (2.0/1.0)
|   |   |   |   |   |   F10 > 2.100000 : 1 (2.0/1.0)
|   |   |   F1 > 1.160000 : 0 (11.0/1.3)
|   F10 > 2.700000 : 1 (42.0/1.4)
