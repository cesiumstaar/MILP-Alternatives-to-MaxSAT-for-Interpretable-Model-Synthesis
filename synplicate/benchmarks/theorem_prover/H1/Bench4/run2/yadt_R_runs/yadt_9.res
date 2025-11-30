YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 30.0613% of 163 cases. 

MISCLASSIFICATION on training: 1.22699% of 163 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	112	2	0
0	0	49	1

MISCLASSIFICATION on test file: 13.5977% of 1059 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	659	111	0
0	33	256	1

DECISION TREE (size=19)

F1 <= -0.250000 : 0 (55.0/1.4)
F1 > -0.250000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (31.0/1.4)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 0.980000 :
|   |   |   |   F1 <= 0.330000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.330000 : 1 (5.0/1.2)
|   |   |   F1 > 0.980000 : 0 (12.0/1.3)
|   F10 > 2.500000 :
|   |   F1 <= 1.950000 :
|   |   |   F1 <= 0.150000 :
|   |   |   |   F10 <= 3.500000 : 0 (3.0/1.1)
|   |   |   |   F10 > 3.500000 : 1 (5.0/1.2)
|   |   |   F1 > 0.150000 : 1 (33.0/1.4)
|   |   F1 > 1.950000 :
|   |   |   F10 <= 3.700000 : 0 (7.0/1.3)
|   |   |   F10 > 3.700000 : 1 (8.0/3.5)
