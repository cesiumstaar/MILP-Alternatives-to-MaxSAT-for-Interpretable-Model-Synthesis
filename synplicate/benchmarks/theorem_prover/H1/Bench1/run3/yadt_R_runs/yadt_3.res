YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 41.1765% of 68 cases. 

MISCLASSIFICATION on training: 0% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	28	0	0
0	0	40	1

MISCLASSIFICATION on test file: 15.5405% of 296 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	111	44	0
0	2	139	1

DECISION TREE (size=13)

F10 <= 1.600000 : 0 (13.0/1.3)
F10 > 1.600000 :
|   F1 <= 0.000000 :
|   |   F1 <= -0.290000 : 0 (9.0/1.3)
|   |   F1 > -0.290000 :
|   |   |   F10 <= 4.100000 : 0 (3.0/1.1)
|   |   |   F10 > 4.100000 : 1 (2.0/1.0)
|   F1 > 0.000000 :
|   |   F10 <= 2.600000 :
|   |   |   F1 <= 1.160000 : 1 (4.0/1.2)
|   |   |   F1 > 1.160000 : 0 (3.0/1.1)
|   |   F10 > 2.600000 : 1 (34.0/1.4)
