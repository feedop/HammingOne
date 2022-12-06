# HammingOne

Autor: Filip Skrzeczkowski

Program w ci¹gu binarnych wektorów dowolnej d³ugoœci, znajduje pary takich, których dystans Hamminga wynosi 1 (ró¿ni¹ siê na dok³adnie jednej pozycji).

## Jak zbudowaæ?

Do plików Ÿród³owych do³¹czony zosta³ Makefile przeznaczony do u¿ytkowania na systemach Unixowych. Po wykonaniu polecenia make w tym folderze (src),
program binarny powinien pokazaæ siê w folderze bin.

W przypadku systemu Windows sugerowanym rozwi¹zaniem jest otworzenie do³¹czonego projektu Visual Studio (Ÿród³owa edycja 2022) i zbudowanie projektu za jego pomoc¹. 
Mo¿na spóbowaæ równie¿ wykorzystaæ narzêdzia implementuj¹ce polecenie make na Windowsie (np. pobrane z choco), choæ nie mo¿na zagwarantowaæ stuprocentowej skutecznoœci
tego rozwi¹zania na ka¿dym komputerze.

## Jak korzystaæ?

Program nale¿y wywo³aæ nastêpum¹co: ./HammingOne [input] [-c] [-v]

Pierwszy argument jest obowi¹zkowy i oznacza plik Ÿród³owy. "-c" i "-v" s¹ opcjonalne i mog¹ zostaæ zamienione kolejnoœci¹.
 * "-c" - odpowiada za wykonanie algorytmu na CPU (oprócz GPU) i wyœwietlenie czasu dzia³ania obu werji
 * "-v" (verbose) - sprawia, ¿e program wyœwietla nie tylko liczbê znalezionych par, ale te¿ wszystkie wektory wchodz¹ce w ich sk³ad.
 Program wyœwietli wszystkie pary, dlatego zalecane jest wykorzystywanie tej opcji tylko w przypadku ma³ych danych.