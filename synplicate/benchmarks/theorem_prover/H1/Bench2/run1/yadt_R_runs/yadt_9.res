YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.5122% of 164 cases. 

MISCLASSIFICATION on training: 1.82927% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	90	1	0
0	2	71	1

MISCLASSIFICATION on test file: 13.81% of 1021 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	565	116	0
0	25	315	1

DECISION TREE (size=19)

F1 <= -0.140000 : 0 (55.0/3.8)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F1 <= 0.920000 :
|   |   |   F1 <= 0.280000 : 0 (4.0/1.2)
|   |   |   F1 > 0.280000 :
|   |   |   |   F10 <= 1.400000 : 0 (2.0/1.0)
|   |   |   |   F10 > 1.400000 : 1 (9.0/1.3)
|   |   F1 > 0.920000 : 0 (23.0/1.3)
|   F10 > 2.700000 :
|   |   F10 <= 4.600000 :
|   |   |   F1 <= 1.880000 :
|   |   |   |   F1 <= -0.120000 : 0 (2.0/1.0)
|   |   |   |   F1 > -0.120000 : 1 (27.0/1.4)
|   |   |   F1 > 1.880000 :
|   |   |   |   F10 <= 3.700000 : 0 (6.0/1.2)
|   |   |   |   F10 > 3.700000 : 1 (4.0/2.2)
|   |   F10 > 4.600000 : 1 (32.0/1.4)
