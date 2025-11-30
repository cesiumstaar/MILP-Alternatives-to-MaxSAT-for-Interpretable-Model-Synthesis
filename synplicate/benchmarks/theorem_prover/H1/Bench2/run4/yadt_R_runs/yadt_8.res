YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.5676% of 148 cases. 

MISCLASSIFICATION on training: 1.35135% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	84	1	0
0	1	62	1

MISCLASSIFICATION on test file: 19.6251% of 907 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	460	155	0
0	23	269	1

DECISION TREE (size=15)

F1 <= -0.220000 : 0 (53.0/1.4)
F1 > -0.220000 :
|   F10 <= 2.800000 :
|   |   F1 <= 1.210000 :
|   |   |   F1 <= 0.280000 : 0 (4.0/1.2)
|   |   |   F1 > 0.280000 :
|   |   |   |   F10 <= 1.300000 : 0 (3.0/1.1)
|   |   |   |   F10 > 1.300000 : 1 (13.0/1.3)
|   |   F1 > 1.210000 : 0 (19.0/1.3)
|   F10 > 2.800000 :
|   |   F1 <= 2.210000 : 1 (45.0/2.6)
|   |   F1 > 2.210000 :
|   |   |   F10 <= 4.400000 : 0 (6.0/2.3)
|   |   |   F10 > 4.400000 : 1 (5.0/1.2)
