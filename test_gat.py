"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
import pytest

from learning.gat import FCGAT
import torch

def test_FCGAT():
	D1 = 100
	D2 = 50
	N = 5
	K = 20
	nn = FCGAT(D1, D2)
	x = torch.randn(N, K, D1)
	x_out = nn(x)

	assert tuple(x_out.shape) == (N, K, D2)
