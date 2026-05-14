# C-MAPSS Data Placement

The repository is organized to keep the NASA C-MAPSS raw files under this directory so the entire project is inspectable and reproducible from raw text data.

```text
data/
  README.md
  CMAPSSData/
    train_FD001.txt
```

The main report uses the `FD001` training trajectories and splits engine units into train, validation, and test subsets inside the code. The robustness appendix also reads `FD002`, `FD003`, and `FD004`.

Expected filename:

- `train_FD001.txt`

Additional files used by the robustness extension:

- `train_FD002.txt`
- `train_FD003.txt`
- `train_FD004.txt`
