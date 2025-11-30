YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.8049% of 164 cases. 

MISCLASSIFICATION on training: 0% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	102	0	0
0	0	62	1

MISCLASSIFICATION on test file: 13.6194% of 536 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	295	71	0
0	2	168	1

DECISION TREE (size=23)

F10 <= 2.100000 :
|   F10 <= 1.500000 : 0 (42.0/1.4)
|   F10 > 1.500000 :
|   |   F1 <= 0.200000 : 0 (11.0/1.3)
|   |   F1 > 0.200000 :
|   |   |   F1 <= 0.990000 : 1 (5.0/1.2)
|   |   |   F1 > 0.990000 : 0 (8.0/1.3)
F10 > 2.100000 :
|   F1 <= -0.010000 :
|   |   F10 <= 4.900000 : 0 (30.0/1.4)
|   |   F10 > 4.900000 : 1 (2.0/1.0)
|   F1 > -0.010000 :
|   |   F10 <= 2.900000 :
|   |   |   F1 <= 1.660000 :
|   |   |   |   F1 <= 0.200000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.200000 :
|   |   |   |   |   F1 <= 1.400000 : 1 (16.0/1.3)
|   |   |   |   |   F1 > 1.400000 :
|   |   |   |   |   |   F10 <= 2.500000 : 0 (2.0/1.0)
|   |   |   |   |   |   F10 > 2.500000 : 1 (3.0/1.1)
|   |   |   F1 > 1.660000 : 0 (7.0/1.3)
|   |   F10 > 2.900000 : 1 (36.0/1.4)
