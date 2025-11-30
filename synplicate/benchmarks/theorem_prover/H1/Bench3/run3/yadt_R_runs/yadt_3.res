YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.2353% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	42	0	0
0	1	25	1

MISCLASSIFICATION on test file: 13.0719% of 306 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	174	31	0
0	9	92	1

DECISION TREE (size=11)

F10 <= 3.000000 :
|   F1 <= 0.380000 : 0 (21.0/1.3)
|   F1 > 0.380000 :
|   |   F1 <= 1.130000 :
|   |   |   F10 <= 1.900000 : 0 (6.0/2.3)
|   |   |   F10 > 1.900000 : 1 (7.0/1.3)
|   |   F1 > 1.130000 : 0 (10.0/1.3)
F10 > 3.000000 :
|   F1 <= -0.060000 : 0 (6.0/1.2)
|   F1 > -0.060000 : 1 (18.0/1.3)
