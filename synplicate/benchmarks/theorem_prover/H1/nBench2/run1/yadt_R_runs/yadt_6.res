YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 41.3793% of 116 cases. 

MISCLASSIFICATION on training: 0% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	68	0	0
0	0	48	1

MISCLASSIFICATION on test file: 19.1638% of 861 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	428	153	0
0	12	268	1

DECISION TREE (size=15)

F1 <= -0.690000 : 0 (40.0/1.4)
F1 > -0.690000 :
|   F10 <= 2.700000 :
|   |   F10 <= 2.100000 : 0 (20.0/1.3)
|   |   F10 > 2.100000 :
|   |   |   F1 <= 0.240000 : 0 (3.0/1.1)
|   |   |   F1 > 0.240000 :
|   |   |   |   F1 <= 1.310000 : 1 (4.0/1.2)
|   |   |   |   F1 > 1.310000 : 0 (3.0/1.1)
|   F10 > 2.700000 :
|   |   F10 <= 3.300000 :
|   |   |   F1 <= 1.310000 : 1 (5.0/1.2)
|   |   |   F1 > 1.310000 : 0 (2.0/1.0)
|   |   F10 > 3.300000 : 1 (39.0/1.4)
