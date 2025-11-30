YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 48.1928% of 83 cases. 

MISCLASSIFICATION on training: 2.40964% of 83 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	38	2	0
0	0	43	1

MISCLASSIFICATION on test file: 18.2635% of 334 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	149	58	0
0	3	124	1

DECISION TREE (size=15)

F1 <= 0.290000 :
|   F10 <= 5.000000 : 0 (23.0/1.3)
|   F10 > 5.000000 : 1 (3.0/2.1)
F1 > 0.290000 :
|   F10 <= 2.600000 :
|   |   F10 <= 1.300000 : 0 (9.0/1.3)
|   |   F10 > 1.300000 :
|   |   |   F1 <= 1.470000 :
|   |   |   |   F10 <= 2.100000 :
|   |   |   |   |   F1 <= 0.720000 : 1 (4.0/2.2)
|   |   |   |   |   F1 > 0.720000 : 0 (2.0/1.0)
|   |   |   |   F10 > 2.100000 : 1 (4.0/1.2)
|   |   |   F1 > 1.470000 : 0 (4.0/1.2)
|   F10 > 2.600000 : 1 (34.0/1.4)
