# C-MAPSS Data Placement

The repository is organized to keep the NASA C-MAPSS raw files under this directory so the entire project is inspectable and reproducible from raw text data.

```text
data/
  README.md
  CMAPSSData/
    train_FD001.txt
    test_FD001.txt
    RUL_FD001.txt
```

The main report uses the `FD001` training trajectories and splits engine units into train, validation, and test subsets inside the code. The robustness appendix also reads `FD002`, `FD003`, and `FD004`.

The repository also includes the official truncated test files and RUL labels. They are useful for standard prediction-only benchmarking, but the main project does not use them as the primary evaluation split because the maintenance-policy experiment requires complete run-to-failure trajectories.

Expected filename:

- `train_FD001.txt`

Additional files used by the robustness extension:

- `train_FD002.txt`
- `train_FD003.txt`
- `train_FD004.txt`
- `test_FD002.txt`
- `test_FD003.txt`
- `test_FD004.txt`
- `RUL_FD002.txt`
- `RUL_FD003.txt`
- `RUL_FD004.txt`
