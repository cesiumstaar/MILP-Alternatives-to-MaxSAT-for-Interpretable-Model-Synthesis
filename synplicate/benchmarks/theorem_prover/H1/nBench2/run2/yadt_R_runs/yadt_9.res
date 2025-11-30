YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 42.0732% of 164 cases. 

MISCLASSIFICATION on training: 1.82927% of 164 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	93	2	0
0	1	68	1

MISCLASSIFICATION on test file: 15.9021% of 981 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	550	128	0
0	28	275	1

DECISION TREE (size=15)

F1 <= -0.140000 : 0 (55.0/2.6)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.900000 : 0 (22.0/1.3)
|   |   F10 > 1.900000 :
|   |   |   F1 <= 1.350000 :
|   |   |   |   F1 <= 0.350000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.350000 : 1 (7.0/1.3)
|   |   |   F1 > 1.350000 : 0 (9.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 1.940000 : 1 (50.0/2.6)
|   |   F1 > 1.940000 :
|   |   |   F10 <= 3.800000 : 0 (6.0/1.2)
|   |   |   F10 > 3.800000 : 1 (13.0/2.5)
