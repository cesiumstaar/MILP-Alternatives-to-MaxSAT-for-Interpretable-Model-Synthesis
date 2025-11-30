YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45.2381% of 84 cases. 

MISCLASSIFICATION on training: 1.19048% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	37	1	0
0	0	46	1

MISCLASSIFICATION on test file: 21.5909% of 352 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	137	69	0
0	7	139	1

DECISION TREE (size=9)

F1 <= 0.100000 : 0 (27.0/1.4)
F1 > 0.100000 :
|   F10 <= 2.600000 :
|   |   F10 <= 1.600000 : 0 (7.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.190000 : 1 (7.0/2.4)
|   |   |   F1 > 1.190000 : 0 (3.0/1.1)
|   F10 > 2.600000 : 1 (40.0/1.4)
