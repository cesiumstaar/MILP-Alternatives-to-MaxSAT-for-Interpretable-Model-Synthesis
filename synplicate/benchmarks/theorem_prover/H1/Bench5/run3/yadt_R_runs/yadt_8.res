YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 41.2162% of 148 cases. 

MISCLASSIFICATION on training: 0.675676% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	86	1	0
0	0	61	1

MISCLASSIFICATION on test file: 15.2439% of 492 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	248	72	0
0	3	169	1

DECISION TREE (size=19)

F10 <= 2.200000 :
|   F10 <= 1.400000 : 0 (39.0/1.4)
|   F10 > 1.400000 :
|   |   F1 <= 0.320000 : 0 (11.0/1.3)
|   |   F1 > 0.320000 :
|   |   |   F1 <= 1.000000 : 1 (6.0/1.2)
|   |   |   F1 > 1.000000 : 0 (8.0/1.3)
F10 > 2.200000 :
|   F1 <= -0.030000 :
|   |   F10 <= 4.900000 : 0 (22.0/1.3)
|   |   F10 > 4.900000 :
|   |   |   F1 <= -0.370000 : 0 (3.0/1.1)
|   |   |   F1 > -0.370000 : 1 (2.0/1.0)
|   F1 > -0.030000 :
|   |   F10 <= 2.900000 :
|   |   |   F1 <= 1.630000 : 1 (10.0/2.4)
|   |   |   F1 > 1.630000 : 0 (3.0/1.1)
|   |   F10 > 2.900000 : 1 (44.0/1.4)
