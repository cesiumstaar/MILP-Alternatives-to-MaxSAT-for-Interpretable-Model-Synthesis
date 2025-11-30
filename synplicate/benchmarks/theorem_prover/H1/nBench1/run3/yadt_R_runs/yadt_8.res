YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.7755% of 147 cases. 

MISCLASSIFICATION on training: 3.40136% of 147 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	89	1	0
0	4	53	1

MISCLASSIFICATION on test file: 19.057% of 509 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	260	83	0
0	14	152	1

DECISION TREE (size=11)

F10 <= 2.200000 : 0 (64.0/6.1)
F10 > 2.200000 :
|   F1 <= -0.090000 :
|   |   F10 <= 5.000000 : 0 (24.0/1.3)
|   |   F10 > 5.000000 : 1 (3.0/2.1)
|   F1 > -0.090000 :
|   |   F10 <= 2.600000 :
|   |   |   F1 <= 1.470000 : 1 (7.0/1.3)
|   |   |   F1 > 1.470000 : 0 (5.0/1.2)
|   |   F10 > 2.600000 : 1 (44.0/1.4)
