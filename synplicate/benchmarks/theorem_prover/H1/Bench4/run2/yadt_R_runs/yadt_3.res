YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.806% of 67 cases. 

MISCLASSIFICATION on training: 2.98507% of 67 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	40	1	0
0	1	25	1

MISCLASSIFICATION on test file: 12.6822% of 686 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	426	58	0
0	29	173	1

DECISION TREE (size=15)

F10 <= 2.900000 :
|   F1 <= 0.300000 : 0 (19.0/1.3)
|   F1 > 0.300000 :
|   |   F1 <= 0.750000 : 1 (4.0/1.2)
|   |   F1 > 0.750000 :
|   |   |   F10 <= 2.000000 : 0 (11.0/1.3)
|   |   |   F10 > 2.000000 :
|   |   |   |   F1 <= 1.110000 : 1 (2.0/1.0)
|   |   |   |   F1 > 1.110000 : 0 (3.0/1.1)
F10 > 2.900000 :
|   F1 <= -0.890000 : 0 (5.0/1.2)
|   F1 > -0.890000 :
|   |   F1 <= 2.110000 : 1 (20.0/2.5)
|   |   F1 > 2.110000 : 0 (3.0/2.1)
