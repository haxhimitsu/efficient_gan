#!/usr/bin/env python3
# coding: UTF-8
import os

class file_control:


    def create_directory(self,directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print("create directory"+directory_path)
        else:
            print("already exist"+directory_path)