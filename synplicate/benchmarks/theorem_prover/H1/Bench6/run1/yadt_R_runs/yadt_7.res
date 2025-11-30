YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 40.9091% of 132 cases. 

MISCLASSIFICATION on training: 1.51515% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	76	2	0
0	0	54	1

MISCLASSIFICATION on test file: 12.0238% of 840 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	496	91	0
0	10	243	1

DECISION TREE (size=19)

F1 <= -0.730000 : 0 (40.0/1.4)
F1 > -0.730000 :
|   F10 <= 2.000000 :
|   |   F10 <= 1.100000 : 0 (16.0/1.3)
|   |   F10 > 1.100000 :
|   |   |   F1 <= 0.780000 :
|   |   |   |   F1 <= 0.270000 : 0 (3.0/1.1)
|   |   |   |   F1 > 0.270000 : 1 (4.0/1.2)
|   |   |   F1 > 0.780000 : 0 (6.0/1.2)
|   F10 > 2.000000 :
|   |   F1 <= -0.230000 :
|   |   |   F10 <= 5.000000 : 0 (6.0/1.2)
|   |   |   F10 > 5.000000 : 1 (3.0/1.1)
|   |   F1 > -0.230000 :
|   |   |   F1 <= 1.750000 : 1 (40.0/2.6)
|   |   |   F1 > 1.750000 :
|   |   |   |   F10 <= 3.000000 : 0 (5.0/1.2)
|   |   |   |   F10 > 3.000000 : 1 (9.0/2.4)
