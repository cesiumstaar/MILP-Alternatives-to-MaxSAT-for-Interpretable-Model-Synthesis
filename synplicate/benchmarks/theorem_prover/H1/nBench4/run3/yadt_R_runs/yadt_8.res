YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 37.1622% of 148 cases. 

MISCLASSIFICATION on training: 0.675676% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	93	0	0
0	1	54	1

MISCLASSIFICATION on test file: 17.6784% of 939 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	505	147	0
0	19	268	1

DECISION TREE (size=15)

F1 <= -0.430000 : 0 (58.0/1.4)
F1 > -0.430000 :
|   F10 <= 2.600000 :
|   |   F1 <= 1.240000 :
|   |   |   F1 <= 0.290000 : 0 (7.0/1.3)
|   |   |   F1 > 0.290000 :
|   |   |   |   F10 <= 1.600000 : 0 (8.0/2.4)
|   |   |   |   F10 > 1.600000 : 1 (6.0/1.2)
|   |   F1 > 1.240000 : 0 (18.0/1.3)
|   F10 > 2.600000 :
|   |   F1 <= 2.240000 : 1 (43.0/1.4)
|   |   F1 > 2.240000 :
|   |   |   F10 <= 3.400000 : 0 (3.0/1.1)
|   |   |   F10 > 3.400000 : 1 (5.0/1.2)
