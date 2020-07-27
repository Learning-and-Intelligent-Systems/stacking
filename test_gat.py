"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
import pytest

from learning.gat import FCGAT
import torch

def test_FCGAT():
	D = 100
	nn = FCGAT(D)
	x = torch.randn(5, 20, D)
	x_out = nn(x)

	assert x.shape == x_out.shape
