import numpy as np
import pytest

from phishing_url.reference import parse_index_text, apply_reference, load_external_index


def test_sql_is_parsed_as_data_without_execution():
    text = "DROP TABLE anything;\n(1, 'https://example.com/a\\\'b', 'a.html', 1, '2021-01-01 00:00:00'),\n(2, 'https://example.com/q?q=hello', 'b.html', 0, '2021-01-01 00:00:00');\n"
    rows = parse_index_text(text)
    assert rows[0]["url"] == "https://example.com/a'b"
    assert rows[1]["label"] == 0


def test_changed_sql_format_is_not_silently_accepted():
    text = "(1, 'https://example.com/', 'a.html', 1, '2021-01-01 00:00:00'),\n(2, BROKEN);\n"
    with pytest.raises(ValueError, match="of 2"):
        parse_index_text(text)


def test_partial_or_modified_external_index_is_rejected(tmp_path):
    path = tmp_path / "index.sql"
    path.write_text("(1, 'https://example.com/', 'a.html', 1, '2021-01-01 00:00:00');\n")
    with pytest.raises(ValueError, match="incomplete"):
        load_external_index(path)


def test_reference_only_changes_exact_matches():
    scores, hits = apply_reference(["https://a.example/", "https://b.example/", "https://a.example/new"],
                                  np.array([.2, .8, .3]), {"https://a.example/": 1})
    assert scores.tolist() == [1, .8, .3]
    assert hits == [0]


def test_indented_sql_records_are_not_lost():
    text="(1, 'https://a.example/', 'a.html', 1, '2021-01-01 00:00:00'),\n  (2, 'https://b.example/', 'b.html', 0, '2021-01-01 00:00:00');\n"
    assert len(parse_index_text(text)) == 2


def test_external_conflicts_are_excluded_and_equal_duplicates_kept(tmp_path):
    path=tmp_path/'index.sql'
    path.write_text("(1, 'https://a.example/', 'a.html', 1, '2021-01-01 00:00:00'),\n(2, 'https://a.example/', 'b.html', 0, '2021-01-01 00:00:00'),\n(3, 'https://b.example/', 'c.html', 1, '2021-01-01 00:00:00'),\n(4, 'https://b.example/', 'd.html', 1, '2021-01-01 00:00:00');\n")
    reference,audit,_=load_external_index(path,verify_hash=False)
    assert reference == {'https://b.example/':1}
    assert audit['conflicting_urls'] == 1
    assert audit['duplicate_records'] == 2
