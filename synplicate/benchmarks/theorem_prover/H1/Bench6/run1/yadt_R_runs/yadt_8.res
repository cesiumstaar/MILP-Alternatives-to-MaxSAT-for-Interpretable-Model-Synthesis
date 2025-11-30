YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.8649% of 148 cases. 

MISCLASSIFICATION on training: 1.35135% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	87	2	0
0	0	59	1

MISCLASSIFICATION on test file: 13.5747% of 884 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	517	107	0
0	13	247	1

DECISION TREE (size=17)

F1 <= -0.730000 : 0 (45.0/1.4)
F1 > -0.730000 :
|   F10 <= 2.700000 :
|   |   F1 <= 0.780000 :
|   |   |   F1 <= 0.270000 : 0 (11.0/1.3)
|   |   |   F1 > 0.270000 : 1 (6.0/1.2)
|   |   F1 > 0.780000 : 0 (19.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= -0.060000 :
|   |   |   F10 <= 5.000000 : 0 (7.0/1.3)
|   |   |   F10 > 5.000000 : 1 (3.0/1.1)
|   |   F1 > -0.060000 :
|   |   |   F1 <= 1.770000 : 1 (41.0/2.6)
|   |   |   F1 > 1.770000 :
|   |   |   |   F10 <= 3.700000 : 0 (5.0/1.2)
|   |   |   |   F10 > 3.700000 : 1 (11.0/2.5)
