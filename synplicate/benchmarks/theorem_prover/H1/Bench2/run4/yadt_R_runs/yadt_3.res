YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.6471% of 68 cases. 

MISCLASSIFICATION on training: 2.94118% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	28	1	0
0	1	38	1

MISCLASSIFICATION on test file: 23.4742% of 639 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	293	123	0
0	27	196	1

DECISION TREE (size=11)

F1 <= -0.220000 : 0 (20.0/1.3)
F1 > -0.220000 :
|   F10 <= 1.300000 : 0 (3.0/1.1)
|   F10 > 1.300000 :
|   |   F1 <= 2.210000 :
|   |   |   F10 <= 2.700000 :
|   |   |   |   F1 <= 1.210000 : 1 (10.0/2.4)
|   |   |   |   F1 > 1.210000 : 0 (2.0/1.0)
|   |   |   F10 > 2.700000 : 1 (29.0/1.4)
|   |   F1 > 2.210000 : 0 (4.0/2.2)
