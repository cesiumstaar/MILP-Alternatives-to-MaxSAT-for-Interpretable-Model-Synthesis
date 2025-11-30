YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.6364% of 132 cases. 

MISCLASSIFICATION on training: 2.27273% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	80	1	0
0	2	49	1

MISCLASSIFICATION on test file: 14.5969% of 459 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	257	59	0
0	8	135	1

DECISION TREE (size=15)

F1 <= 0.170000 : 0 (48.0/3.8)
F1 > 0.170000 :
|   F10 <= 2.800000 :
|   |   F10 <= 1.600000 : 0 (21.0/1.3)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.410000 :
|   |   |   |   F1 <= 0.280000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.280000 :
|   |   |   |   |   F10 <= 2.100000 :
|   |   |   |   |   |   F1 <= 1.000000 : 1 (3.0/1.1)
|   |   |   |   |   |   F1 > 1.000000 : 0 (2.0/1.0)
|   |   |   |   |   F10 > 2.100000 : 1 (10.0/1.3)
|   |   |   F1 > 1.410000 : 0 (8.0/1.3)
|   F10 > 2.800000 : 1 (37.0/2.6)
