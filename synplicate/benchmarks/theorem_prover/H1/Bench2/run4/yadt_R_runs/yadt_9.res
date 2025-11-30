YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.6829% of 164 cases. 

MISCLASSIFICATION on training: 1.21951% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	92	2	0
0	0	70	1

MISCLASSIFICATION on test file: 19.4652% of 935 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	473	161	0
0	21	280	1

DECISION TREE (size=15)

F1 <= -0.220000 : 0 (55.0/1.4)
F1 > -0.220000 :
|   F10 <= 2.800000 :
|   |   F1 <= 1.210000 :
|   |   |   F10 <= 1.300000 : 0 (5.0/1.2)
|   |   |   F10 > 1.300000 :
|   |   |   |   F1 <= 0.280000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.280000 : 1 (14.0/1.3)
|   |   F1 > 1.210000 : 0 (23.0/1.3)
|   F10 > 2.800000 :
|   |   F1 <= 2.210000 : 1 (50.0/2.6)
|   |   F1 > 2.210000 :
|   |   |   F10 <= 4.000000 : 0 (6.0/1.2)
|   |   |   F10 > 4.000000 : 1 (8.0/2.4)
