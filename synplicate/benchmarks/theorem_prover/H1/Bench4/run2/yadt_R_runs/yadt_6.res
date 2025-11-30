YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 34.7826% of 115 cases. 

MISCLASSIFICATION on training: 0.869565% of 115 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	75	0	0
0	1	39	1

MISCLASSIFICATION on test file: 11.6173% of 878 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	554	68	0
0	34	222	1

DECISION TREE (size=19)

F1 <= -0.490000 : 0 (35.0/1.4)
F1 > -0.490000 :
|   F10 <= 2.500000 :
|   |   F10 <= 1.500000 : 0 (19.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 0.980000 :
|   |   |   |   F1 <= 0.330000 : 0 (4.0/1.2)
|   |   |   |   F1 > 0.330000 : 1 (5.0/1.2)
|   |   |   F1 > 0.980000 : 0 (8.0/1.3)
|   F10 > 2.500000 :
|   |   F1 <= 1.950000 :
|   |   |   F1 <= 0.110000 :
|   |   |   |   F10 <= 3.700000 : 0 (2.0/1.0)
|   |   |   |   F10 > 3.700000 : 1 (3.0/1.1)
|   |   |   F1 > 0.110000 : 1 (28.0/1.4)
|   |   F1 > 1.950000 :
|   |   |   F10 <= 4.500000 : 0 (8.0/2.4)
|   |   |   F10 > 4.500000 : 1 (3.0/1.1)
