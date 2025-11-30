YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 36.5854% of 164 cases. 

MISCLASSIFICATION on training: 3.04878% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	103	1	0
0	4	56	1

MISCLASSIFICATION on test file: 17.301% of 578 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	320	77	0
0	23	158	1

DECISION TREE (size=11)

F1 <= 0.210000 : 0 (66.0/5.0)
F1 > 0.210000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.300000 : 0 (23.0/1.3)
|   |   F10 > 1.300000 :
|   |   |   F1 <= 1.000000 :
|   |   |   |   F1 <= 0.310000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.310000 : 1 (11.0/2.5)
|   |   |   F1 > 1.000000 : 0 (16.0/2.5)
|   F10 > 2.700000 : 1 (46.0/1.4)
