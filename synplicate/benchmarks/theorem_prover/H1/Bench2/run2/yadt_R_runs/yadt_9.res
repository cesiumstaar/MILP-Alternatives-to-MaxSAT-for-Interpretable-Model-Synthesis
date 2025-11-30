YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 31.0976% of 164 cases. 

MISCLASSIFICATION on training: 3.65854% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	112	1	0
0	5	46	1

MISCLASSIFICATION on test file: 12.5793% of 946 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	623	83	0
0	36	204	1

DECISION TREE (size=17)

F10 <= 2.000000 : 0 (69.0/6.1)
F10 > 2.000000 :
|   F1 <= 0.150000 :
|   |   F10 <= 5.200000 : 0 (31.0/1.4)
|   |   F10 > 5.200000 :
|   |   |   F1 <= -0.770000 : 0 (2.0/1.0)
|   |   |   F1 > -0.770000 : 1 (2.0/1.0)
|   F1 > 0.150000 :
|   |   F1 <= 1.740000 :
|   |   |   F10 <= 2.700000 :
|   |   |   |   F1 <= 0.340000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.340000 : 1 (11.0/2.5)
|   |   |   F10 > 2.700000 : 1 (28.0/1.4)
|   |   F1 > 1.740000 :
|   |   |   F10 <= 3.700000 : 0 (13.0/2.5)
|   |   |   F10 > 3.700000 : 1 (6.0/1.2)
