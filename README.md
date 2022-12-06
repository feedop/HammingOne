# HammingOne

Autor: Filip Skrzeczkowski

Program w ci�gu binarnych wektor�w dowolnej d�ugo�ci, znajduje pary takich, kt�rych dystans Hamminga wynosi 1 (r�ni� si� na dok�adnie jednej pozycji).

## Jak zbudowa�?

Do plik�w �r�d�owych do��czony zosta� Makefile przeznaczony do u�ytkowania na systemach Unixowych. Po wykonaniu polecenia make w tym folderze (src),
program binarny powinien pokaza� si� w folderze bin.

W przypadku systemu Windows sugerowanym rozwi�zaniem jest otworzenie do��czonego projektu Visual Studio (�r�d�owa edycja 2022) i zbudowanie projektu za jego pomoc�. 
Mo�na sp�bowa� r�wnie� wykorzysta� narz�dzia implementuj�ce polecenie make na Windowsie (np. pobrane z choco), cho� nie mo�na zagwarantowa� stuprocentowej skuteczno�ci
tego rozwi�zania na ka�dym komputerze.

## Jak korzysta�?

Program nale�y wywo�a� nast�pum�co: ./HammingOne [input] [-c] [-v]

Pierwszy argument jest obowi�zkowy i oznacza plik �r�d�owy. "-c" i "-v" s� opcjonalne i mog� zosta� zamienione kolejno�ci�.
 * "-c" - odpowiada za wykonanie algorytmu na CPU (opr�cz GPU) i wy�wietlenie czasu dzia�ania obu werji
 * "-v" (verbose) - sprawia, �e program wy�wietla nie tylko liczb� znalezionych par, ale te� wszystkie wektory wchodz�ce w ich sk�ad.
 Program wy�wietli wszystkie pary, dlatego zalecane jest wykorzystywanie tej opcji tylko w przypadku ma�ych danych.