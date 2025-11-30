YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 48.8095% of 84 cases. 

MISCLASSIFICATION on training: 0% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	41	0	0
0	0	43	1

MISCLASSIFICATION on test file: 23.6544% of 706 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	319	153	0
0	14	220	1

DECISION TREE (size=13)

F1 <= 0.100000 : 0 (31.0/1.4)
F1 > 0.100000 :
|   F10 <= 2.300000 :
|   |   F1 <= 0.320000 : 0 (4.0/1.2)
|   |   F1 > 0.320000 :
|   |   |   F1 <= 0.890000 : 1 (4.0/1.2)
|   |   |   F1 > 0.890000 : 0 (3.0/1.1)
|   F10 > 2.300000 :
|   |   F1 <= 1.600000 : 1 (33.0/1.4)
|   |   F1 > 1.600000 :
|   |   |   F10 <= 3.900000 : 0 (3.0/1.1)
|   |   |   F10 > 3.900000 : 1 (6.0/1.2)
