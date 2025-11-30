YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 33.1081% of 148 cases. 

MISCLASSIFICATION on training: 2.02703% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	98	1	0
0	2	47	1

MISCLASSIFICATION on test file: 15.3518% of 938 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	581	106	0
0	38	213	1

DECISION TREE (size=13)

F1 <= 0.160000 : 0 (60.0/2.6)
F1 > 0.160000 :
|   F10 <= 2.300000 :
|   |   F1 <= 0.780000 :
|   |   |   F10 <= 1.100000 : 0 (6.0/1.2)
|   |   |   F10 > 1.100000 : 1 (9.0/2.4)
|   |   F1 > 0.780000 : 0 (25.0/1.3)
|   F10 > 2.300000 :
|   |   F1 <= 1.630000 : 1 (33.0/1.4)
|   |   F1 > 1.630000 :
|   |   |   F10 <= 3.900000 : 0 (9.0/2.4)
|   |   |   F10 > 3.900000 : 1 (6.0/1.2)
