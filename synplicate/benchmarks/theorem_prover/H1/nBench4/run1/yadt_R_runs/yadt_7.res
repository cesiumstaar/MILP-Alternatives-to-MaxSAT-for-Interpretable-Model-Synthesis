YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.3939% of 132 cases. 

MISCLASSIFICATION on training: 1.51515% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	79	1	0
0	1	51	1

MISCLASSIFICATION on test file: 11.88% of 867 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	515	83	0
0	20	249	1

DECISION TREE (size=17)

F1 <= 0.150000 :
|   F10 <= 5.000000 : 0 (43.0/1.4)
|   F10 > 5.000000 :
|   |   F1 <= -0.580000 : 0 (6.0/1.2)
|   |   F1 > -0.580000 : 1 (2.0/1.0)
F1 > 0.150000 :
|   F10 <= 1.400000 : 0 (20.0/1.3)
|   F10 > 1.400000 :
|   |   F10 <= 4.000000 :
|   |   |   F1 <= 1.850000 :
|   |   |   |   F10 <= 2.400000 :
|   |   |   |   |   F1 <= 0.850000 : 1 (8.0/2.4)
|   |   |   |   |   F1 > 0.850000 : 0 (3.0/1.1)
|   |   |   |   F10 > 2.400000 : 1 (17.0/1.3)
|   |   |   F1 > 1.850000 : 0 (8.0/2.4)
|   |   F10 > 4.000000 : 1 (25.0/1.3)
