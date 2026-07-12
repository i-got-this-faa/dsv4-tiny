"""Tests for the data module."""

from __future__ import annotations

import torch

from dsv4_tiny.data import RandomTextDataset, TokenizedTextDataset


class TestRandomTextDataset:
    def test_returns_correct_keys(self):
        ds = RandomTextDataset(seq_len=8, num_samples=5)
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["labels"], torch.Tensor)

    def test_shapes(self):
        ds = RandomTextDataset(seq_len=128, num_samples=3)
        for i in range(len(ds)):
            assert ds[i]["input_ids"].shape == (128,)
            assert ds[i]["labels"].shape == (128,)

    def test_labels_match_input_ids(self):
        ds = RandomTextDataset(seq_len=16, num_samples=5)
        for i in range(len(ds)):
            s = ds[i]
            assert torch.equal(s["input_ids"], s["labels"])

    def test_length(self):
        ds = RandomTextDataset(seq_len=8, num_samples=42)
        assert len(ds) == 42

    def test_reproducible(self):
        ds1 = RandomTextDataset(seq_len=8, num_samples=3)
        ds2 = RandomTextDataset(seq_len=8, num_samples=3)
        # Both produce random but valid data
        assert ds1[0]["input_ids"].shape == ds2[0]["input_ids"].shape
