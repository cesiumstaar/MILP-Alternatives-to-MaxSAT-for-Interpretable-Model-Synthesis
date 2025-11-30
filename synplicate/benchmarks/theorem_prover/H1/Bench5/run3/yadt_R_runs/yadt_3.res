YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 50% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	33	1	0
0	0	34	1

MISCLASSIFICATION on test file: 17.5497% of 302 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	143	47	0
0	6	106	1

DECISION TREE (size=11)

F1 <= 0.270000 :
|   F10 <= 4.600000 : 0 (22.0/1.3)
|   F10 > 4.600000 : 1 (3.0/2.1)
F1 > 0.270000 :
|   F10 <= 1.400000 : 0 (7.0/1.3)
|   F10 > 1.400000 :
|   |   F1 <= 1.320000 : 1 (26.0/1.3)
|   |   F1 > 1.320000 :
|   |   |   F10 <= 2.900000 : 0 (4.0/1.2)
|   |   |   F10 > 2.900000 : 1 (6.0/1.2)
