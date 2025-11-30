YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.1176% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	29	1	0
0	0	38	1

MISCLASSIFICATION on test file: 24.8494% of 664 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	292	152	0
0	13	207	1

DECISION TREE (size=11)

F1 <= 0.100000 : 0 (22.0/1.3)
F1 > 0.100000 :
|   F10 <= 3.900000 :
|   |   F1 <= 1.600000 :
|   |   |   F10 <= 2.300000 :
|   |   |   |   F1 <= 0.320000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.320000 : 1 (5.0/2.3)
|   |   |   F10 > 2.300000 : 1 (13.0/1.3)
|   |   F1 > 1.600000 : 0 (4.0/1.2)
|   F10 > 3.900000 : 1 (21.0/1.3)
