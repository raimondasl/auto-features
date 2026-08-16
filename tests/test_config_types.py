"""Config leaves are type-checked at load, so YAML's footguns fail by name.

Two of them have already fired in this project:

* **A quoted number.** `w_embedding: "1.5"` parses as a string, and the string reached
  `validate_config`'s bounds check, which raised `TypeError: '<' not supported between
  instances of 'str' and 'int'`. Not from `rr update` — from *every* command that loads a
  config, with a traceback and no mention of the field at fault. Under `rr watch` it took
  the whole loop down, because `watch_loop` does not catch it.
* **An unquoted `off`.** PyYAML implements YAML 1.1, where `off`/`no`/`yes` are booleans,
  so `enrichment: provider: off` set `provider = False`. `_normalize_off` already handles
  that one, and these tests pin that the new checking does not undo it.

The check is derived from `dataclasses.fields()` rather than a list of fields to watch,
for the reason C-16 records: a hand-listed subset checks whatever somebody remembered and
reads exactly like one that checks everything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reporadar.config import RepoRadarConfig, load_config, validate_config


def _load(tmp_path: Path, yml: str) -> Any:
    (tmp_path / ".reporadar.yml").write_text(yml, encoding="utf-8")
    return load_config(tmp_path / ".reporadar.yml")


class TestQuotedNumbersAreAccepted:
    """One unambiguous meaning, so coerced rather than refused."""

    def test_quoted_float(self, tmp_path: Path) -> None:
        cfg = _load(tmp_path, 'repo_path: .\nranking:\n  w_embedding: "1.5"\n')
        assert cfg.ranking.w_embedding == 1.5
        assert isinstance(cfg.ranking.w_embedding, float)

    def test_quoted_int(self, tmp_path: Path) -> None:
        cfg = _load(tmp_path, 'repo_path: .\narxiv:\n  lookback_days: "14"\n')
        assert cfg.arxiv.lookback_days == 14
        assert isinstance(cfg.arxiv.lookback_days, int)

    def test_an_int_is_fine_where_a_float_is_declared(self, tmp_path: Path) -> None:
        cfg = _load(tmp_path, "repo_path: .\nranking:\n  w_embedding: 2\n")
        assert cfg.ranking.w_embedding == 2.0

    def test_nested_sections_are_checked_too(self, tmp_path: Path) -> None:
        """`triage.finescale` is the one section a top-level walk does not reach."""
        cfg = _load(tmp_path, 'repo_path: .\ntriage:\n  finescale:\n    threshold: "0.67"\n')
        assert cfg.triage.finescale.threshold == pytest.approx(0.67)

    def test_the_original_crash_is_gone(self, tmp_path: Path) -> None:
        """The regression this exists for: `validate_config` raised TypeError, from every
        command, on a config a user could plausibly write."""
        cfg = _load(
            tmp_path,
            'repo_path: .\narxiv:\n  lookback_days: "14"\n  max_results_per_query: "50"\n',
        )
        assert validate_config(cfg) == []


class TestBooleanStringsAreNotTruthiness:
    """The trap this nearly shipped with: `bool("false")` is True, so a quoted off-switch
    would have turned the stage ON — silently, and worse than the crash being fixed."""

    def test_quoted_false_is_false(self, tmp_path: Path) -> None:
        cfg = _load(tmp_path, 'repo_path: .\ntriage:\n  enabled: "false"\n')
        assert cfg.triage.enabled is False

    def test_quoted_true_is_true(self, tmp_path: Path) -> None:
        cfg = _load(tmp_path, 'repo_path: .\ntriage:\n  enabled: "true"\n')
        assert cfg.triage.enabled is True

    def test_a_word_that_is_not_a_boolean_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="expects a boolean"):
            _load(tmp_path, 'repo_path: .\ntriage:\n  enabled: "maybe"\n')


class TestTheOffSwitchStillWorks:
    """`_normalize_off` predates this check and must survive it: `provider: off` is the
    documented form, and YAML hands it over as the boolean False."""

    def test_unquoted_off(self, tmp_path: Path) -> None:
        cfg = _load(tmp_path, "repo_path: .\nenrichment:\n  provider: off\n")
        assert cfg.enrichment.provider == "off"

    def test_quoted_off(self, tmp_path: Path) -> None:
        cfg = _load(tmp_path, 'repo_path: .\nenrichment:\n  provider: "off"\n')
        assert cfg.enrichment.provider == "off"

    def test_the_normaliser_is_reached_through_the_registry(self) -> None:
        """A guard against the fix drifting: the check runs BEFORE the dataclass is built,
        so a leaf with its own meaning for a bool has to be registered or it gets refused."""
        from reporadar.config import _NORMALIZERS, _normalize_off

        assert _NORMALIZERS["enrichment.provider"] is _normalize_off


class TestWhatIsRefusedAndHowItReads:
    def test_a_bool_where_a_number_belongs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"ranking\.w_keyword expects float"):
            _load(tmp_path, "repo_path: .\nranking:\n  w_keyword: true\n")

    def test_a_word_where_a_number_belongs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a number"):
            _load(tmp_path, "repo_path: .\nranking:\n  w_embedding: abc\n")

    def test_the_message_names_the_field(self, tmp_path: Path) -> None:
        """The point of the change. The old failure said `'<' not supported between
        instances of 'str' and 'int'` and left the user to find the field themselves."""
        with pytest.raises(ValueError) as exc:
            _load(tmp_path, "repo_path: .\nranking:\n  w_embedding: abc\n")
        assert "ranking.w_embedding" in str(exc.value)

    def test_a_string_section_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="expects a mapping"):
            _load(tmp_path, "repo_path: .\nranking: not-a-mapping\n")

    def test_yaml_bool_for_a_string_field_explains_the_gotcha(self, tmp_path: Path) -> None:
        """A string field with no registered normaliser: the message has to teach the
        YAML 1.1 rule, because nothing about `sort_by: on` looks wrong to a reader."""
        with pytest.raises(ValueError, match="YAML 1.1"):
            _load(tmp_path, "repo_path: .\narxiv:\n  sort_by: on\n")


class TestItDoesNotGetInTheWay:
    def test_lists_and_dicts_pass_through(self, tmp_path: Path) -> None:
        cfg = _load(
            tmp_path,
            "repo_path: .\narxiv:\n  categories: [cs.LG, cs.CL]\n"
            "ranking:\n  category_weights:\n    cs.LG: 1.5\n",
        )
        assert cfg.arxiv.categories == ["cs.LG", "cs.CL"]
        assert cfg.ranking.category_weights == {"cs.LG": 1.5}

    def test_both_shipped_templates_still_load(self, tmp_path: Path) -> None:
        """The check runs on every load, so a template it rejects would break `rr init`
        for everyone — the loudest possible regression, and worth pinning directly."""
        from reporadar.config import default_config_yaml, measured_config_yaml

        for text in (default_config_yaml(), measured_config_yaml()):
            cfg = _load(tmp_path, text)
            assert isinstance(cfg, RepoRadarConfig)

    def test_an_unknown_key_still_reports_itself(self, tmp_path: Path) -> None:
        """Unknown keys pass through so the dataclass raises its own error, which already
        names the key — re-implementing that here would be a second message to keep true."""
        with pytest.raises(TypeError, match="no_such_field"):
            _load(tmp_path, "repo_path: .\nranking:\n  no_such_field: 1\n")


def test_every_section_is_covered_not_a_remembered_subset() -> None:
    """The C-16 property: the section map is derived from `RepoRadarConfig`'s own fields,
    so adding a section cannot leave it unchecked."""
    from dataclasses import fields, is_dataclass

    from reporadar.config import _section_types

    expected = {
        f.name
        for f in fields(RepoRadarConfig)
        if isinstance(f.default_factory, type) and is_dataclass(f.default_factory)  # type: ignore[misc]
    }
    assert set(_section_types()) == expected
    assert len(expected) >= 15, "the walk found suspiciously few sections"
