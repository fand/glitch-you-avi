#!/bin/bash

g++ $(PKG_CONFIG_PATH=/usr/local/opt/opencv@2/lib/pkgconfig pkg-config --cflags --libs opencv) avi2.cc -o glitch
