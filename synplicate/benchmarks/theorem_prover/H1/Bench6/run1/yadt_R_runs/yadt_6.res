YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.2414% of 116 cases. 

MISCLASSIFICATION on training: 0.862069% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	66	1	0
0	0	49	1

MISCLASSIFICATION on test file: 12.0603% of 796 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	471	80	0
0	16	229	1

DECISION TREE (size=21)

F1 <= -0.730000 : 0 (35.0/1.4)
F1 > -0.730000 :
|   F10 <= 2.000000 :
|   |   F10 <= 1.100000 : 0 (16.0/1.3)
|   |   F10 > 1.100000 :
|   |   |   F1 <= 0.610000 :
|   |   |   |   F1 <= 0.270000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.270000 : 1 (3.0/1.1)
|   |   |   F1 > 0.610000 : 0 (5.0/1.2)
|   F10 > 2.000000 :
|   |   F10 <= 3.000000 :
|   |   |   F1 <= 1.750000 :
|   |   |   |   F1 <= 0.180000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.180000 : 1 (10.0/1.3)
|   |   |   F1 > 1.750000 : 0 (4.0/1.2)
|   |   F10 > 3.000000 :
|   |   |   F1 <= -0.250000 :
|   |   |   |   F10 <= 5.000000 : 0 (2.0/1.0)
|   |   |   |   F10 > 5.000000 : 1 (2.0/1.0)
|   |   |   F1 > -0.250000 : 1 (35.0/2.6)
