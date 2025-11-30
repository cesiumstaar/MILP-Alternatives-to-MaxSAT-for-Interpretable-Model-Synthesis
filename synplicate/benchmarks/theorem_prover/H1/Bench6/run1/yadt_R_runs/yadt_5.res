YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	62	1	0
0	0	37	1

MISCLASSIFICATION on test file: 12.4346% of 764 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	455	81	0
0	14	214	1

DECISION TREE (size=19)

F1 <= -0.730000 : 0 (33.0/1.4)
F1 > -0.730000 :
|   F10 <= 2.000000 :
|   |   F10 <= 1.100000 : 0 (16.0/1.3)
|   |   F10 > 1.100000 :
|   |   |   F1 <= 0.610000 :
|   |   |   |   F1 <= 0.270000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.270000 : 1 (3.0/1.1)
|   |   |   F1 > 0.610000 : 0 (5.0/1.2)
|   F10 > 2.000000 :
|   |   F1 <= 1.750000 :
|   |   |   F1 <= -0.250000 :
|   |   |   |   F10 <= 5.000000 : 0 (3.0/1.1)
|   |   |   |   F10 > 5.000000 : 1 (2.0/1.0)
|   |   |   F1 > -0.250000 : 1 (28.0/1.4)
|   |   F1 > 1.750000 :
|   |   |   F10 <= 3.000000 : 0 (3.0/1.1)
|   |   |   F10 > 3.000000 : 1 (5.0/2.3)
