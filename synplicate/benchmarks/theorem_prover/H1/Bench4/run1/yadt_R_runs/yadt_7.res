YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 24.2424% of 132 cases. 

MISCLASSIFICATION on training: 5.30303% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	100	0	0
0	7	25	1

MISCLASSIFICATION on test file: 15.7489% of 908 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	617	46	0
0	97	148	1

DECISION TREE (size=11)

F10 <= 2.300000 : 0 (76.0/7.3)
F10 > 2.300000 :
|   F1 <= 0.120000 : 0 (23.0/2.5)
|   F1 > 0.120000 :
|   |   F1 <= 1.950000 :
|   |   |   F10 <= 3.000000 :
|   |   |   |   F1 <= 0.410000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.410000 : 1 (3.0/1.1)
|   |   |   F10 > 3.000000 : 1 (22.0/1.3)
|   |   F1 > 1.950000 : 0 (6.0/2.3)
