YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 24.3902% of 164 cases. 

MISCLASSIFICATION on training: 1.21951% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	123	1	0
0	1	39	1

MISCLASSIFICATION on test file: 12.7992% of 961 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	660	71	0
0	52	178	1

DECISION TREE (size=11)

F10 <= 1.700000 : 0 (69.0/1.4)
F10 > 1.700000 :
|   F1 <= -0.140000 : 0 (34.0/1.4)
|   F1 > -0.140000 :
|   |   F1 <= 1.250000 : 1 (31.0/2.6)
|   |   F1 > 1.250000 :
|   |   |   F10 <= 2.900000 : 0 (15.0/1.3)
|   |   |   F10 > 2.900000 :
|   |   |   |   F1 <= 2.250000 : 1 (9.0/1.3)
|   |   |   |   F1 > 2.250000 : 0 (6.0/2.3)
