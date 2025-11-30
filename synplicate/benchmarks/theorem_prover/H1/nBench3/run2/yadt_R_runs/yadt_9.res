YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.0244% of 164 cases. 

MISCLASSIFICATION on training: 1.21951% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	98	2	0
0	0	64	1

MISCLASSIFICATION on test file: 17.8967% of 542 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	279	89	0
0	8	166	1

DECISION TREE (size=17)

F1 <= 0.120000 :
|   F10 <= 4.500000 : 0 (51.0/1.4)
|   F10 > 4.500000 :
|   |   F1 <= -0.310000 : 0 (6.0/1.2)
|   |   F1 > -0.310000 : 1 (2.0/1.0)
F1 > 0.120000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.600000 : 0 (29.0/1.4)
|   |   F10 > 1.600000 :
|   |   |   F1 <= 1.220000 : 1 (13.0/3.6)
|   |   |   F1 > 1.220000 : 0 (10.0/1.3)
|   F10 > 2.500000 :
|   |   F1 <= 1.690000 : 1 (46.0/1.4)
|   |   F1 > 1.690000 :
|   |   |   F10 <= 3.100000 : 0 (2.0/1.0)
|   |   |   F10 > 3.100000 : 1 (5.0/1.2)
