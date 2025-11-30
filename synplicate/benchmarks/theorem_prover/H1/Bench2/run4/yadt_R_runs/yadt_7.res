YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45.4545% of 132 cases. 

MISCLASSIFICATION on training: 2.27273% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	71	1	0
0	2	58	1

MISCLASSIFICATION on test file: 19.788% of 849 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	425	142	0
0	26	256	1

DECISION TREE (size=13)

F1 <= -0.120000 : 0 (50.0/2.6)
F1 > -0.120000 :
|   F10 <= 2.700000 :
|   |   F1 <= 1.210000 :
|   |   |   F10 <= 1.300000 : 0 (3.0/1.1)
|   |   |   F10 > 1.300000 : 1 (13.0/2.5)
|   |   F1 > 1.210000 : 0 (15.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 2.210000 : 1 (42.0/1.4)
|   |   F1 > 2.210000 :
|   |   |   F10 <= 4.400000 : 0 (5.0/2.3)
|   |   |   F10 > 4.400000 : 1 (4.0/1.2)
