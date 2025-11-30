YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 32.5758% of 132 cases. 

MISCLASSIFICATION on training: 5.30303% of 132 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	87	2	0
0	5	38	1

MISCLASSIFICATION on test file: 12.0231% of 865 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	572	68	0
0	36	189	1

DECISION TREE (size=13)

F10 <= 2.000000 : 0 (60.0/6.1)
F10 > 2.000000 :
|   F1 <= 0.150000 :
|   |   F10 <= 5.000000 : 0 (23.0/1.3)
|   |   F10 > 5.000000 :
|   |   |   F1 <= -0.790000 : 0 (2.0/1.0)
|   |   |   F1 > -0.790000 : 1 (2.0/1.0)
|   F1 > 0.150000 :
|   |   F1 <= 1.740000 : 1 (34.0/3.8)
|   |   F1 > 1.740000 :
|   |   |   F10 <= 3.700000 : 0 (7.0/2.4)
|   |   |   F10 > 3.700000 : 1 (4.0/1.2)
