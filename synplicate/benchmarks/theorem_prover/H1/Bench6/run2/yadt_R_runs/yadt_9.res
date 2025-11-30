YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 40.8537% of 164 cases. 

MISCLASSIFICATION on training: 1.82927% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	96	1	0
0	2	65	1

MISCLASSIFICATION on test file: 13.5841% of 957 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	558	99	0
0	31	269	1

DECISION TREE (size=15)

F1 <= -0.260000 : 0 (55.0/2.6)
F1 > -0.260000 :
|   F10 <= 1.900000 : 0 (26.0/2.6)
|   F10 > 1.900000 :
|   |   F10 <= 3.800000 :
|   |   |   F1 <= 2.000000 :
|   |   |   |   F1 <= 0.190000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.190000 :
|   |   |   |   |   F1 <= 1.340000 : 1 (24.0/1.3)
|   |   |   |   |   F1 > 1.340000 :
|   |   |   |   |   |   F10 <= 3.000000 : 0 (3.0/1.1)
|   |   |   |   |   |   F10 > 3.000000 : 1 (3.0/1.1)
|   |   |   F1 > 2.000000 : 0 (11.0/1.3)
|   |   F10 > 3.800000 : 1 (39.0/2.6)
