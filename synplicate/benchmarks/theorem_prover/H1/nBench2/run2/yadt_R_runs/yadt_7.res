YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 43.9394% of 132 cases. 

MISCLASSIFICATION on training: 2.27273% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	73	1	0
0	2	56	1

MISCLASSIFICATION on test file: 17.8322% of 858 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	468	121	0
0	32	237	1

DECISION TREE (size=15)

F1 <= -0.140000 : 0 (45.0/2.6)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.900000 : 0 (15.0/1.3)
|   |   F10 > 1.900000 :
|   |   |   F1 <= 0.960000 :
|   |   |   |   F1 <= 0.350000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.350000 : 1 (6.0/1.2)
|   |   |   F1 > 0.960000 : 0 (7.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 1.940000 : 1 (43.0/2.6)
|   |   F1 > 1.940000 :
|   |   |   F10 <= 4.400000 : 0 (6.0/2.3)
|   |   |   F10 > 4.400000 : 1 (8.0/1.3)
