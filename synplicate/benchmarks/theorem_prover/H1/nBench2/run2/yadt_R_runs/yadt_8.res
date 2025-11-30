YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.5676% of 148 cases. 

MISCLASSIFICATION on training: 2.02703% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	83	2	0
0	1	62	1

MISCLASSIFICATION on test file: 15.9067% of 943 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	530	122	0
0	28	263	1

DECISION TREE (size=15)

F1 <= -0.140000 : 0 (53.0/2.6)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.900000 : 0 (16.0/1.3)
|   |   F10 > 1.900000 :
|   |   |   F1 <= 0.960000 :
|   |   |   |   F1 <= 0.350000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.350000 : 1 (6.0/1.2)
|   |   |   F1 > 0.960000 : 0 (8.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 1.940000 : 1 (46.0/2.6)
|   |   F1 > 1.940000 :
|   |   |   F10 <= 3.800000 : 0 (5.0/1.2)
|   |   |   F10 > 3.800000 : 1 (12.0/2.5)
