YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.4242% of 132 cases. 

MISCLASSIFICATION on training: 1.51515% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	76	0	0
0	2	54	1

MISCLASSIFICATION on test file: 18.1818% of 451 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	241	56	0
0	26	128	1

DECISION TREE (size=17)

F1 <= 0.200000 :
|   F10 <= 3.500000 : 0 (36.0/1.4)
|   F10 > 3.500000 :
|   |   F10 <= 4.900000 :
|   |   |   F1 <= -0.020000 : 0 (10.0/1.3)
|   |   |   F1 > -0.020000 : 1 (2.0/1.0)
|   |   F10 > 4.900000 : 1 (2.0/1.0)
F1 > 0.200000 :
|   F10 <= 2.800000 :
|   |   F10 <= 1.500000 : 0 (17.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 0.990000 : 1 (11.0/1.3)
|   |   |   F1 > 0.990000 : 0 (12.0/2.5)
|   F10 > 2.800000 :
|   |   F1 <= 1.870000 : 1 (39.0/1.4)
|   |   F1 > 1.870000 : 0 (3.0/2.1)
