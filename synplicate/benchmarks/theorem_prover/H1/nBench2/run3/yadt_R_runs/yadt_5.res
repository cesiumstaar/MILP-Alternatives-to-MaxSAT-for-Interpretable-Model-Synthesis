YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 48% of 100 cases. 

MISCLASSIFICATION on training: 1% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	51	1	0
0	0	48	1

MISCLASSIFICATION on test file: 19.2602% of 784 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	385	138	0
0	13	248	1

DECISION TREE (size=13)

F1 <= 0.100000 : 0 (40.0/1.4)
F1 > 0.100000 :
|   F10 <= 3.000000 :
|   |   F1 <= 1.600000 :
|   |   |   F10 <= 2.300000 :
|   |   |   |   F1 <= 0.320000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.320000 :
|   |   |   |   |   F1 <= 0.890000 : 1 (4.0/1.2)
|   |   |   |   |   F1 > 0.890000 : 0 (2.0/1.0)
|   |   |   F10 > 2.300000 : 1 (9.0/1.3)
|   |   F1 > 1.600000 : 0 (5.0/1.2)
|   F10 > 3.000000 : 1 (36.0/2.6)
