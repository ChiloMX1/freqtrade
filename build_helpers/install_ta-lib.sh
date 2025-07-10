#!/bin/bash
set -e

echo "Instalando TA-Lib C 0.6.4..."

cd /tmp
tar -xvzf ta-lib-0.6.4-src.tar.gz
cd ta-lib-0.6.4
./configure --prefix=/usr/local
make
make install

echo "TA-Lib 0.6.4 instalado correctamente."
