YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.7647% of 68 cases. 

MISCLASSIFICATION on training: 1.47059% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	43	0	0
0	1	24	1

MISCLASSIFICATION on test file: 11.9497% of 318 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	189	24	0
0	14	91	1

DECISION TREE (size=11)

F10 <= 1.600000 : 0 (20.0/1.3)
F10 > 1.600000 :
|   F1 <= -0.030000 : 0 (12.0/1.3)
|   F1 > -0.030000 :
|   |   F10 <= 3.100000 :
|   |   |   F1 <= 0.390000 : 0 (5.0/1.2)
|   |   |   F1 > 0.390000 :
|   |   |   |   F1 <= 1.170000 : 1 (9.0/1.3)
|   |   |   |   F1 > 1.170000 : 0 (7.0/2.4)
|   |   F10 > 3.100000 : 1 (15.0/1.3)
