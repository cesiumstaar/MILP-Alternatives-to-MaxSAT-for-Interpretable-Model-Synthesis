YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 29.932% of 147 cases. 

MISCLASSIFICATION on training: 2.72109% of 147 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	102	1	0
0	3	41	1

MISCLASSIFICATION on test file: 13.1451% of 1027 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	656	88	0
0	47	236	1

DECISION TREE (size=17)

F1 <= -0.250000 : 0 (53.0/1.4)
F1 > -0.250000 :
|   F10 <= 2.000000 : 0 (39.0/4.9)
|   F10 > 2.000000 :
|   |   F1 <= 1.950000 :
|   |   |   F10 <= 3.500000 :
|   |   |   |   F1 <= 0.230000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.230000 : 1 (14.0/2.5)
|   |   |   F10 > 3.500000 : 1 (23.0/1.3)
|   |   F1 > 1.950000 :
|   |   |   F10 <= 3.700000 : 0 (7.0/1.3)
|   |   |   F10 > 3.700000 :
|   |   |   |   F10 <= 4.500000 :
|   |   |   |   |   F1 <= 2.150000 : 1 (2.0/1.0)
|   |   |   |   |   F1 > 2.150000 : 0 (2.0/1.0)
|   |   |   |   F10 > 4.500000 : 1 (3.0/1.1)
