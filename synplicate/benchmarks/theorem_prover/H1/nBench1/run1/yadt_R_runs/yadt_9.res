YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.8049% of 164 cases. 

MISCLASSIFICATION on training: 1.21951% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	100	2	0
0	0	62	1

MISCLASSIFICATION on test file: 16.934% of 561 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	300	88	0
0	7	166	1

DECISION TREE (size=13)

F1 <= -0.100000 : 0 (54.0/1.4)
F1 > -0.100000 :
|   F10 <= 2.500000 :
|   |   F1 <= 0.900000 :
|   |   |   F10 <= 2.100000 :
|   |   |   |   F10 <= 1.300000 : 0 (12.0/1.3)
|   |   |   |   F10 > 1.300000 :
|   |   |   |   |   F1 <= 0.440000 : 0 (6.0/1.2)
|   |   |   |   |   F1 > 0.440000 : 1 (5.0/2.3)
|   |   |   F10 > 2.100000 : 1 (7.0/1.3)
|   |   F1 > 0.900000 : 0 (28.0/1.4)
|   F10 > 2.500000 : 1 (52.0/2.6)
