"""Tests for reporadar.source_analysis."""

from __future__ import annotations

from pathlib import Path

from reporadar.source_analysis import (
    detect_ml_patterns,
    extract_identifiers,
    extract_imports,
)


class TestExtractImports:
    def test_python_imports(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text(
            "import torch\nimport numpy as np\nfrom transformers import AutoModel\n",
            encoding="utf-8",
        )
        imports = extract_imports(tmp_path, extensions=[".py"])
        assert "torch" in imports
        assert "numpy" in imports
        assert "transformers" in imports

    def test_skips_stdlib(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text(
            "import os\nimport sys\nimport json\nimport requests\n",
            encoding="utf-8",
        )
        imports = extract_imports(tmp_path, extensions=[".py"])
        assert "os" not in imports
        assert "sys" not in imports
        assert "json" not in imports
        assert "requests" in imports

    def test_js_imports(self, tmp_path: Path) -> None:
        (tmp_path / "app.js").write_text(
            'import React from "react";\n'
            'const express = require("express");\n',
            encoding="utf-8",
        )
        imports = extract_imports(tmp_path, extensions=[".js"])
        assert "react" in imports
        assert "express" in imports

    def test_ts_imports(self, tmp_path: Path) -> None:
        (tmp_path / "app.ts").write_text(
            'import axios from "axios";\n',
            encoding="utf-8",
        )
        imports = extract_imports(tmp_path, extensions=[".ts"])
        assert "axios" in imports

    def test_skips_git_dir(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git" / "hooks"
        git_dir.mkdir(parents=True)
        (git_dir / "pre-commit.py").write_text("import secretlib\n", encoding="utf-8")
        (tmp_path / "main.py").write_text("import torch\n", encoding="utf-8")
        imports = extract_imports(tmp_path, extensions=[".py"])
        assert "secretlib" not in imports
        assert "torch" in imports

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        nm = tmp_path / "node_modules" / "foo"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text('import bar from "bar";\n', encoding="utf-8")
        imports = extract_imports(tmp_path, extensions=[".js"])
        assert "bar" not in imports

    def test_max_files_limit(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f"mod{i}.py").write_text(f"import pkg{i}\n", encoding="utf-8")
        imports = extract_imports(tmp_path, extensions=[".py"], max_files=5)
        # Should have at most 5 unique imports (from 5 files)
        assert len(imports) <= 5

    def test_empty_repo(self, tmp_path: Path) -> None:
        imports = extract_imports(tmp_path)
        assert imports == []

    def test_skips_relative_js_imports(self, tmp_path: Path) -> None:
        (tmp_path / "app.js").write_text(
            'import foo from "./foo";\nimport bar from "bar";\n',
            encoding="utf-8",
        )
        imports = extract_imports(tmp_path, extensions=[".js"])
        assert "bar" in imports
        # Relative imports should be skipped
        assert not any("foo" in i for i in imports)


class TestDetectMlPatterns:
    def test_detects_backward(self, tmp_path: Path) -> None:
        (tmp_path / "train.py").write_text("loss.backward()\n", encoding="utf-8")
        signals = detect_ml_patterns(tmp_path)
        assert "autograd/backprop" in signals

    def test_detects_nn_module(self, tmp_path: Path) -> None:
        (tmp_path / "model.py").write_text(
            "class MyModel(nn.Module):\n    pass\n", encoding="utf-8"
        )
        signals = detect_ml_patterns(tmp_path)
        assert "PyTorch neural networks" in signals

    def test_detects_dataloader(self, tmp_path: Path) -> None:
        (tmp_path / "data.py").write_text(
            "loader = DataLoader(dataset, batch_size=32)\n", encoding="utf-8"
        )
        signals = detect_ml_patterns(tmp_path)
        assert "data loading pipeline" in signals

    def test_no_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "hello.py").write_text("print('hello')\n", encoding="utf-8")
        signals = detect_ml_patterns(tmp_path)
        assert signals == []

    def test_empty_repo(self, tmp_path: Path) -> None:
        signals = detect_ml_patterns(tmp_path)
        assert signals == []

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "train.py").write_text(
            "class Net(nn.Module):\n    pass\n"
            "loader = DataLoader(ds)\n"
            "loss.backward()\n"
            "optimizer.step()\n",
            encoding="utf-8",
        )
        signals = detect_ml_patterns(tmp_path)
        assert len(signals) >= 3


class TestExtractIdentifiers:
    def test_extracts_classes(self, tmp_path: Path) -> None:
        (tmp_path / "models.py").write_text(
            "class MyModel:\n    pass\n\nclass _Internal:\n    pass\n",
            encoding="utf-8",
        )
        ids = extract_identifiers(tmp_path)
        assert "MyModel" in ids
        # Private classes should be skipped
        assert "_Internal" not in ids

    def test_extracts_functions(self, tmp_path: Path) -> None:
        (tmp_path / "utils.py").write_text(
            "def train_model():\n    pass\n\ndef _helper():\n    pass\n",
            encoding="utf-8",
        )
        ids = extract_identifiers(tmp_path)
        assert "train_model" in ids
        assert "_helper" not in ids

    def test_skips_syntax_errors(self, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("def foo(\n", encoding="utf-8")
        (tmp_path / "good.py").write_text("def bar():\n    pass\n", encoding="utf-8")
        ids = extract_identifiers(tmp_path)
        assert "bar" in ids

    def test_empty_repo(self, tmp_path: Path) -> None:
        ids = extract_identifiers(tmp_path)
        assert ids == []
