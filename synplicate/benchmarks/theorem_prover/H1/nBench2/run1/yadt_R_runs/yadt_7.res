YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.8788% of 132 cases. 

MISCLASSIFICATION on training: 0% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	82	0	0
0	0	50	1

MISCLASSIFICATION on test file: 16.0347% of 923 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	495	136	0
0	12	280	1

DECISION TREE (size=15)

F1 <= -0.690000 : 0 (50.0/1.4)
F1 > -0.690000 :
|   F10 <= 2.700000 :
|   |   F10 <= 2.100000 : 0 (24.0/1.3)
|   |   F10 > 2.100000 :
|   |   |   F1 <= 0.240000 : 0 (3.0/1.1)
|   |   |   F1 > 0.240000 :
|   |   |   |   F1 <= 1.310000 : 1 (4.0/1.2)
|   |   |   |   F1 > 1.310000 : 0 (3.0/1.1)
|   F10 > 2.700000 :
|   |   F10 <= 3.300000 :
|   |   |   F1 <= 1.310000 : 1 (5.0/1.2)
|   |   |   F1 > 1.310000 : 0 (2.0/1.0)
|   |   F10 > 3.300000 : 1 (41.0/1.4)
