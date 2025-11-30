YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 38.1679% of 131 cases. 

MISCLASSIFICATION on training: 0% of 131 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	81	0	0
0	0	50	1

MISCLASSIFICATION on test file: 12.1749% of 846 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	505	91	0
0	12	238	1

DECISION TREE (size=19)

F1 <= -0.280000 : 0 (45.0/1.4)
F1 > -0.280000 :
|   F10 <= 2.600000 :
|   |   F1 <= 1.190000 :
|   |   |   F10 <= 1.200000 : 0 (8.0/1.3)
|   |   |   F10 > 1.200000 :
|   |   |   |   F1 <= 0.280000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.280000 : 1 (10.0/1.3)
|   |   F1 > 1.190000 : 0 (20.0/1.3)
|   F10 > 2.600000 :
|   |   F1 <= 1.890000 :
|   |   |   F1 <= -0.110000 :
|   |   |   |   F10 <= 4.600000 : 0 (2.0/1.0)
|   |   |   |   F10 > 4.600000 : 1 (2.0/1.0)
|   |   |   F1 > -0.110000 : 1 (33.0/1.4)
|   |   F1 > 1.890000 :
|   |   |   F10 <= 3.700000 : 0 (4.0/1.2)
|   |   |   F10 > 3.700000 : 1 (5.0/1.2)
