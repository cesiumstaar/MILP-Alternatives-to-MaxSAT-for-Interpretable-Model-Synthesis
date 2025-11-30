YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 44.697% of 132 cases. 

MISCLASSIFICATION on training: 3.0303% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	72	1	0
0	3	56	1

MISCLASSIFICATION on test file: 16.0754% of 902 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	485	116	0
0	29	272	1

DECISION TREE (size=17)

F1 <= -0.140000 : 0 (45.0/3.8)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F1 <= 0.920000 :
|   |   |   F1 <= 0.400000 : 0 (4.0/1.2)
|   |   |   F1 > 0.400000 : 1 (9.0/2.4)
|   |   F1 > 0.920000 : 0 (18.0/1.3)
|   F10 > 2.700000 :
|   |   F1 <= 1.880000 :
|   |   |   F1 <= -0.120000 :
|   |   |   |   F10 <= 5.100000 : 0 (2.0/1.0)
|   |   |   |   F10 > 5.100000 : 1 (2.0/1.0)
|   |   |   F1 > -0.120000 : 1 (40.0/1.4)
|   |   F1 > 1.880000 :
|   |   |   F10 <= 4.100000 : 0 (6.0/2.3)
|   |   |   F10 > 4.100000 : 1 (6.0/1.2)
