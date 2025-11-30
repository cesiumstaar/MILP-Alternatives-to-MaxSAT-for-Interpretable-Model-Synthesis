YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 46.4286% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	44	1	0
0	0	39	1

MISCLASSIFICATION on test file: 18.3473% of 714 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	374	109	0
0	22	209	1

DECISION TREE (size=15)

F1 <= -0.140000 : 0 (25.0/1.3)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.900000 : 0 (10.0/1.3)
|   |   F10 > 1.900000 :
|   |   |   F1 <= 1.350000 :
|   |   |   |   F1 <= 0.350000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.350000 : 1 (4.0/1.2)
|   |   |   F1 > 1.350000 : 0 (4.0/1.2)
|   F10 > 2.700000 :
|   |   F1 <= 1.940000 : 1 (33.0/2.6)
|   |   F1 > 1.940000 :
|   |   |   F10 <= 4.400000 : 0 (3.0/1.1)
|   |   |   F10 > 4.400000 : 1 (3.0/1.1)
