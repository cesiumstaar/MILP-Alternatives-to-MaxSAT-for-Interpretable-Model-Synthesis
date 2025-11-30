YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.2414% of 116 cases. 

MISCLASSIFICATION on training: 0.862069% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	48	1	0
0	0	67	1

MISCLASSIFICATION on test file: 17.1913% of 413 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	152	65	0
0	6	190	1

DECISION TREE (size=13)

F1 <= -0.290000 : 0 (24.0/1.3)
F1 > -0.290000 :
|   F10 <= 1.600000 : 0 (13.0/1.3)
|   F10 > 1.600000 :
|   |   F1 <= 0.000000 :
|   |   |   F10 <= 4.300000 : 0 (6.0/1.2)
|   |   |   F10 > 4.300000 : 1 (5.0/2.3)
|   |   F1 > 0.000000 :
|   |   |   F10 <= 2.900000 :
|   |   |   |   F1 <= 1.330000 : 1 (8.0/1.3)
|   |   |   |   F1 > 1.330000 : 0 (5.0/1.2)
|   |   |   F10 > 2.900000 : 1 (55.0/1.4)
