YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.5135% of 148 cases. 

MISCLASSIFICATION on training: 2.02703% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	90	1	0
0	2	55	1

MISCLASSIFICATION on test file: 15.1394% of 502 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	282	67	0
0	9	144	1

DECISION TREE (size=15)

F1 <= 0.170000 : 0 (54.0/3.8)
F1 > 0.170000 :
|   F10 <= 2.900000 :
|   |   F10 <= 1.600000 : 0 (23.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.410000 :
|   |   |   |   F1 <= 0.280000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.280000 :
|   |   |   |   |   F10 <= 2.100000 :
|   |   |   |   |   |   F1 <= 1.000000 : 1 (3.0/1.1)
|   |   |   |   |   |   F1 > 1.000000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 2.100000 : 1 (11.0/1.3)
|   |   |   F1 > 1.410000 : 0 (10.0/1.3)
|   F10 > 2.900000 : 1 (42.0/2.6)
