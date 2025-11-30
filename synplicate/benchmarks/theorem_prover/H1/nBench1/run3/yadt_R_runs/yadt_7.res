YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.7481% of 131 cases. 

MISCLASSIFICATION on training: 3.05344% of 131 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	74	1	0
0	3	53	1

MISCLASSIFICATION on test file: 20.8791% of 455 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	214	82	0
0	13	146	1

DECISION TREE (size=11)

F10 <= 2.100000 : 0 (48.0/5.0)
F10 > 2.100000 :
|   F1 <= -0.090000 :
|   |   F10 <= 5.000000 : 0 (24.0/1.3)
|   |   F10 > 5.000000 : 1 (3.0/2.1)
|   F1 > -0.090000 :
|   |   F10 <= 2.600000 :
|   |   |   F1 <= 1.470000 : 1 (7.0/1.3)
|   |   |   F1 > 1.470000 : 0 (5.0/1.2)
|   |   F10 > 2.600000 : 1 (44.0/1.4)
